import os
import json
import time
import torch
from prompt import Prompt
from generation import Generation
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel


class SampleRewrite:
    def __init__(self, task, root_path:str, llm_root_path:str, llm_name:str, llm_dict:dict, mode:str="base", adapter_path:str=None):
        self.task = task
        self.root_path = root_path
        self.cur_path = root_path + "/inference_control"
        self.llm_root_path = llm_root_path
        self.llm_path = f"{llm_root_path}/{llm_name}"
        self.llm_name = llm_name
        self.llm_dict = llm_dict
        self.mode = mode
        self.adapter_path = adapter_path
        assert mode in ["base", "sft"]
        if self.mode == "sft" and self.adapter_path is None:
            raise Exception("adapter_path must be specified when using sft")

        self.s_llm_name = self.llm_dict[llm_name]
        self.processor, self.model = self._load_lllm()
        if self.mode == "sft":
            self._load_lora_adapter(self.adapter_path)
        self.prompt_builder_class = Prompt()
        self.generator_class = Generation(self.model, self.processor)

    def _load_lllm(self):
        processor = AutoProcessor.from_pretrained(
            self.llm_path,
            local_files_only=True
        )
        # tokenizer = processor.tokenizer

        model = AutoModelForCausalLM.from_pretrained(
            self.llm_path,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )
        print(getattr(model, "hf_device_map", "hf_device_map is not available"))
        return processor, model

    def _load_lora_adapter(self, adapter_path):
        """Load a trained LoRA adapter onto the base model."""

        if self.model is None:
            raise RuntimeError("Base model must be loaded first.")

        self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)

        self.model.eval()

        print(f"LoRA adapter loaded: {adapter_path}")

        return self.model

    def generate_sample_rewrite_candidates(
            self, sample, task, direction, num_candidates=20, max_length_ratio=1.3, min_max_new_tokens=32,
            temperature=0.95, do_sample=True, num_beams=1):
        """
        Generate multiple full-sample rewrite candidates.

        The same sample prompt is duplicated num_candidates times,
        and each prompt independently generates one candidate.

        Parameters
        ----------
        sample : str
            Original complete sample.

        task : str
            "yelp" or "cnndm".

        direction : str
            Yelp:
                "positive" / "negative"

            CNN/DM:
                "simpler" / "complex"

        num_candidates : int
            Number of independently generated sample-level candidates.

        max_length_ratio : float
            Maximum generation length relative to source token length.

        min_max_new_tokens : int
            Minimum max_new_tokens.

        temperature : float
            Sampling temperature.

        do_sample : bool
            Whether to use sampling.

        num_beams : int
            Beam size when sampling is disabled.

        batch_size : int
            Batch size used by generate_batch().

        Returns
        -------
        list[dict]
            A list containing one sample-rewrite record.
        """

        # 1. Build prompt
        user_prompt = (self.prompt_builder_class.build_sample_rewrite_prompt(
                sample=sample, task=task, direction=direction))

        # 2. Calculate source token length
        source_tokens = (self.generator_class.tokenizer.encode(
                sample, add_special_tokens=False))

        source_token_length = len(source_tokens)

        max_new_tokens = max(min_max_new_tokens, int(source_token_length * max_length_ratio))

        # 3. Generate n independent sampled candidates
        outputs = (
            self.generator_class
            .generate(
                prompt=user_prompt,
                num_return_sequences=num_candidates,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                num_beams=num_beams
            )
        )

        # 4. Build result
        result = {
            "task": task,
            "direction": direction,
            "mode": f"{self.mode}_sample_rewrite",
            "source_sample": sample,
            "user_prompt": user_prompt,
            "raw_output": [],
            "source_token_length": source_token_length,
            "max_new_tokens": max_new_tokens
        }

        # 5. Parse generated candidates
        for raw_output in outputs:
            parsed = raw_output.strip()
            if len(parsed) == 0:
                continue
            result["raw_output"].append(raw_output)

        return [result]

    def candidate_out(self, data_path, num_candidates=20, max_length_ratio=1.3, temperature=0.95, do_sample=True,
            num_beams=1):
        """
        Generate sample-level rewrite candidates.
        Each complete source sample is rewritten directly without
        sentence segmentation.
        Yelp:
            level 0 -> positive
            level 1~3 -> positive / negative
            level 4 -> negative
        CNN/DM:
            level 0 -> simpler
            level 1~2 -> simpler / complex
            level 3 -> complex
        """

        # 1. Load dataset
        with open(data_path, "r", encoding="utf-8") as f:
            dataset_test_data = json.load(f)

        task = "cnndm" if "cnn_dm" in data_path else "yelp"
        max_level = 4 if task == "cnndm" else 5

        # 2. Save path
        save_path = f"{self.cur_path}/{task}_{self.s_llm_name}_{self.mode}_sample_rewrite.json"

        # If already generated, do not overwrite
        if os.path.exists(save_path):
            print(f"Sample rewrite file already exists: {save_path}")
            return None

        # 3. Result structure
        # [
        #     [candidate_direction_1, candidate_direction_2],
        #     [candidate_direction_1, candidate_direction_2],
        #     ...
        # ]
        # Each outer element corresponds to one source sample.
        all_results = []
        # 4. Iterate style levels
        for style_level in range(max_level):
            print(f"\n========== style_level: {style_level} ==========")

            # 5. Determine valid directions
            if task == "cnndm":
                directions = ["simpler", "complex"]
                if style_level == 0:
                    directions = ["simpler"]
                elif style_level == 3:
                    directions = ["complex"]
            elif task == "yelp":
                directions = ["positive", "negative"]
                if style_level == 0:
                    directions = ["positive"]
                elif style_level == 4:
                    directions = ["negative"]
            else:
                raise ValueError("task must be 'cnndm' or 'yelp'.")

            # 6. Iterate samples
            for sample_id, sample in enumerate(dataset_test_data[str(style_level)]):
                # Only use first 300 samples per style level
                max_sample = 300 if self.mode == "base" else 100
                if sample_id >= max_sample:
                    break
                sample_start_time = time.time()
                sample_results = []

                # 7. Generate each valid transfer direction
                for direction in directions:
                    try:
                        rewrite_results = (
                            self.generate_sample_rewrite_candidates(
                                sample=sample,
                                task=task,
                                direction=direction,
                                num_candidates=num_candidates,
                                max_length_ratio=max_length_ratio,
                                temperature=temperature,
                                do_sample=do_sample,
                                num_beams=num_beams
                            )
                        )
                        # Add dataset-level identity information
                        for result in rewrite_results:
                            result["style_level"] = style_level
                            result["sample_id"] = sample_id
                        sample_results.extend(rewrite_results)
                    except Exception as e:
                        print("****** sample rewrite error ******")
                        print(f"style_level={style_level}, sample_id={sample_id}, direction={direction}")
                        print(e)
                        print("------------- error -------------")
                        continue

                # 8. Store results for this sample
                all_results.append(sample_results)

                # 9. Timing
                sample_end_time = time.time()
                candidate_num = sum(
                    len(result.get("restored_sample", []))
                    for result in sample_results
                )

                print(
                    f"sample_id:{sample_id}, "
                    f"directions:{len(directions)}, "
                    f"candidates:{candidate_num}, "
                    f"spend_time:"
                    f"{(sample_end_time - sample_start_time) / 60:.2f} min"
                )

                # 10. Incremental saving
                # Sample rewrite can be expensive, therefore save after each sample.
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)

        return None



if __name__ == "__main__":
    mode = "sft"
    root_path = r"F:/python_code/StyleTrans2"
    llm_root_path = r"F:/python_code/pre_model"
    # root_path = r"/root/private_data/StyleTrans2"
    # llm_root_path = r"/root/private_data/"
    llm_name = "gemma-4-E2B-it"
    llm_dict = {"gemma-4-12B-it": "ge12B",
                "gemma-4-E2B-it": "geE2B",
                "Phi-4-mini-instruct": "phi4mini",
                "Phi-4": "phi4"}

    data_file_path = f"{root_path}/data/yelp_test.json"
    task = "cnndm" if "cnn_dm" in data_file_path else "yelp"

    if mode == "base":
        sr_class = SampleRewrite(task, root_path, llm_root_path, llm_name, llm_dict)
        sr_class.candidate_out(data_file_path)
    else:
        adapter_path = root_path + f"/sft/outputs/{task}_{llm_dict[llm_name]}_sft/final_adapter"
        print(os.path.exists(adapter_path))
        sr_class = SampleRewrite(task, root_path, llm_root_path, llm_name, llm_dict,
                                 mode=mode, adapter_path=adapter_path)
        sr_class.candidate_out(data_file_path)


