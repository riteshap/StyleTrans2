
import os
import re
import json
import torch
import time
from transformers import AutoProcessor, AutoModelForCausalLM

from constituency import Constituency
from prompt import Prompt
from generation import Generation


class CandidateOut:
    def __init__(self, root_path:str, llm_root_path:str, llm_name:str, llm_dict:dict):
        self.root_path = root_path
        self.cur_path = root_path + "/inference_control"
        self.llm_root_path = llm_root_path
        self.llm_path = f"{llm_root_path}/{llm_name}"
        self.llm_name = llm_name
        self.llm_dict = llm_dict
        self.s_llm_name = self.llm_dict[llm_name]
        self.processor, self.model = self._load_lllm()
        self.constituency_class = Constituency()
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
        print(
            getattr(
                model,
                "hf_device_map",
                "hf_device_map is not available"
            )
        )
        return processor, model

    def print_gpu_memory(self, tag=""):

        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        peak_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
        free_mem, total_mem = torch.cuda.mem_get_info()
        free_mem /= 1024 ** 3
        total_mem /= 1024 ** 3

        print(
            f"[{tag}] "
            f"allocated={allocated:.2f} GB, "
            f"reserved={reserved:.2f} GB, "
            f"peak_allocated={peak_allocated:.2f} GB, "
            f"peak_reserved={peak_reserved:.2f} GB, "
            f"free={free_mem:.2f}/{total_mem:.2f} GB"
        )

    def split_sentences(self, text: str, dataset: str):
        """
        Split text into sentences according to dataset-specific rules.
        Parameters
        ----------
        text : str  Original text.
        dataset : str  "cnndm" or "yelp"
        Returns : list[str]  List of sentences.
        """
        dataset = dataset.lower()

        if dataset == "cnndm":
            # CNN/DM uses actual newline characters
            sentences = text.split("\n")
            sentences = [s.strip() for s in sentences if s.strip()]

            return sentences

        elif dataset == "yelp":
            # Yelp contains literal "\n"
            paragraphs = text.split("\\n")
            sentences = []
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                # Split only when:
                # 1. punctuation is . ? !
                # 2. punctuation is preceded by an English letter
                # 3. punctuation is followed by whitespace
                parts = re.split(
                    r'(?<=[A-Za-z][.!?])\s+',
                    paragraph
                )
                for part in parts:
                    part = part.strip()
                    if part:
                        sentences.append(part)
            return sentences
        else:
            raise ValueError("dataset must be 'cnndm' or 'yelp'.")

    def test_fill_mode(self, sentences, task, direction, include_blank_info, style_guided, num_return_sequences=3,
            max_new_tokens=8, temperature=0.95, do_sample=True, num_beams=1, max_words=4, batch_size=64):
        """
        Sample-level batch generation for Fill / Replace.
        Parameters
        ----------
        sentences : list[str]
            All sentences belonging to one sample.
        Returns
        -------
        list[dict]
            Same candidate-level output structure as before.
        """

        results = []
        # 1. Define mode
        if style_guided:
            mode = "replace" if include_blank_info else "fill"
        else:
            mode = "plain_replace" if include_blank_info else "plain_fill"

        # 2. Build a sample-level generation queue
        jobs = []
        prompts = []
        for sent_id, sentence in enumerate(sentences):

            # Skip invalid sentence before LLM generation
            try:
                candidates = (
                    self.constituency_class.get_mask_candidates(sentence, max_words=max_words)
                )
            except Exception as e:
                print(f"[Skip sentence] sent_id={sent_id}, error={e}")
                continue

            if len(candidates) == 0:
                continue

            # Build prompts for this sentence
            for candidate_id, candidate in enumerate(candidates):
                if style_guided:
                    user_prompt = (
                        self.prompt_builder_class
                        .build_fill_prompt(
                            candidate=candidate,
                            task=task,
                            direction=direction,
                            include_blank_info=
                            include_blank_info
                        )
                    )
                else:
                    user_prompt = (
                        self.prompt_builder_class
                        .build_plain_fill_prompt(
                            candidate=candidate,
                            include_blank_info=
                            include_blank_info
                        )
                    )

                # prompt queue
                prompts.append(user_prompt)
                # metadata queue
                jobs.append(
                    {
                        "sent_id": sent_id,
                        "sentence": sentence,
                        "candidate_id": candidate_id,
                        "candidate": candidate,
                        "user_prompt": user_prompt
                    }
                )

        # No valid local edits in this sample
        if len(prompts) == 0:
            return []

        print("test_fill_mode.sample prompt count:", len(prompts))

        # 3. SAMPLE-LEVEL batch generation
        batch_outputs = (self.generator_class.generate_batch(
                prompts=prompts,
                batch_size=batch_size,
                num_return_sequences=num_return_sequences,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                num_beams=num_beams
            )
        )

        if len(batch_outputs) != len(jobs):
            raise RuntimeError(f"Generation output mismatch: {len(batch_outputs)} != {len(jobs)}")

        # 4. Restore candidate-level results
        for job, outputs in zip(jobs, batch_outputs):
            sent_id = job["sent_id"]
            sentence = job["sentence"]
            candidate_id = (job["candidate_id"])
            candidate = (job["candidate"])
            user_prompt = (job["user_prompt"])
            result = {
                "task": task,
                "sent_id": sent_id,
                "direction": direction,
                "mode": mode,
                "source_sentence": sentence,
                "start_char": candidate["start_char"],
                "end_char": candidate["end_char"],
                "candidate_id": candidate_id,
                "label": candidate.get("label"),
                "original_span": candidate.get("text"),
                "pos_tags": candidate.get("pos_tags"),
                "masked_text": candidate.get("masked_text"),
                "user_prompt": user_prompt,
                "raw_output": [],
                "parsed": [],
                "generated_span": [],
                "restored_sentence": []
            }

            # Save generated candidates
            for raw_output in outputs:
                parsed = raw_output.strip()
                generated_span = parsed
                restored_sentence = (candidate["left"] + generated_span + candidate["right"])
                result["raw_output"].append(raw_output)
                result["parsed"].append(parsed)
                result["generated_span"].append(generated_span)
                result["restored_sentence"].append(restored_sentence)
            results.append(result)

        return results

    def test_rewrite_mode(self, sentence, task, direction, num_return_sequences=3, max_length_ratio=1.5,
                          min_max_new_tokens=12, temperature=0.95, do_sample=True, num_beams=1):
        """
        Test full-sentence style rewriting.
        """

        # 1. Build normal user instruction
        user_prompt = self.prompt_builder_class.build_rewrite_prompt(
            sentence=sentence,
            task=task,
            direction=direction
        )

        # 2. Calculate source token length
        source_tokens = self.generator_class.tokenizer.encode(
            sentence,
            add_special_tokens=False
        )

        source_token_length = len(source_tokens)

        max_new_tokens = max(
            min_max_new_tokens,
            int(source_token_length * max_length_ratio)
        )

        # 3. Generate
        # Generation internally applies Gemma-IT chat template
        outputs = self.generator_class.generate(
            prompt=user_prompt,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams
        )

        result = {
            "task": task,
            "direction": direction,
            "mode": "rewrite",
            "source_sentence": sentence,
            "candidate_id": None,
            "label": None,
            "original_span": None,
            "pos_tags": None,
            "masked_text": None,
            "user_prompt": user_prompt,
            "raw_output": [],
            "parsed": [],
            "generated_span": [],
            "restored_sentence": [],
            "source_token_length": source_token_length,
            "max_new_tokens": max_new_tokens
        }
        # 4. Save results
        for generation_id, raw_output in enumerate(outputs):
            parsed = raw_output.strip()
            generated_span = parsed
            restored_sentence = parsed
            result["raw_output"].append(raw_output)

        return [result]

    def test_sample(self, sentences, task, style_level: int, num_return_sequences=3, fill_max_new_tokens=8,
            rewrite_max_length_ratio=1.5, temperature=0.9, do_sample=True, num_beams=1, max_words=4,
            fill_batch_size=64):
        """
        Run Fill / Replace / Rewrite generation for one complete sample.
        Parameters
        ----------
        sentences : list[str]
            All sentences belonging to one sample.
        task : str
            "cnndm" or "yelp"
        style_level : int
            Current source style level.
            CNN/DM: 0 ~ 3
            Yelp: 0 ~ 4
        num_return_sequences : int
            Number of generated candidates for each generation prompt.
        fill_max_new_tokens : int
            Maximum generation length for Fill / Replace.
        rewrite_max_length_ratio : float
            Rewrite generation length ratio.
        temperature : float
            Sampling temperature.
        do_sample : bool
            Whether to use sampling.
        num_beams : int
            Beam number when beam search decoding is used.
        max_words : int
            Maximum number of words allowed in one local
            syntactic editing unit.
        fill_batch_size : int
            Batch size for sample-level Fill / Replace generation.
        Returns
        -------
        list[dict]
            All Fill / Replace / Rewrite results of this sample.
        """

        task = task.lower()
        # 1. Determine valid transfer directions
        if task == "cnndm":
            directions = ["simpler", "complex"]
            # level 0: FRE < 40 only move toward simpler
            if style_level == 0:
                directions = ["simpler"]
            # level 3: FRE 80~100 only move toward complex
            elif style_level == 3:
                directions = ["complex"]
        elif task == "yelp":
            directions = ["positive", "negative"]
            # lowest sentiment level
            if style_level == 0:
                directions = ["positive"]
            # highest sentiment level
            elif style_level == 4:
                directions = ["negative"]
        else:
            raise ValueError("task must be 'cnndm' or 'yelp'.")

        # 2. Basic input check
        if not isinstance(sentences, list):
            raise TypeError("sentences must be a list[str].")

        if len(sentences) == 0:
            return []

        all_results = []

        # 3. Run each valid direction
        for direction in directions:

            # 3.1 Fill
            # All sentences in the sample are processed together through sample-level batch generation.
            fill_results = (
                self.test_fill_mode(
                    sentences=sentences,
                    task=task,
                    direction=direction,
                    include_blank_info=False,
                    style_guided=True,
                    num_return_sequences=num_return_sequences,
                    max_new_tokens=fill_max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    max_words=max_words,
                    batch_size=fill_batch_size
                )
            )
            all_results.extend(fill_results)

            # 3.2 Replace
            # Same sample-level batch strategy as Fill.
            replace_results = (
                self.test_fill_mode(
                    sentences=sentences,
                    task=task,
                    direction=direction,
                    include_blank_info=True,
                    style_guided=True,
                    num_return_sequences=num_return_sequences,
                    max_new_tokens=fill_max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    max_words=max_words,
                    batch_size=fill_batch_size
                )
            )
            all_results.extend(replace_results)

            # 3.3 Rewrite
            # Rewrite remains sentence-level because every sentence has its own dynamic max_new_tokens.
            for sent_id, sentence in enumerate(sentences):
                if not isinstance(sentence, str):
                    continue
                sentence = sentence.strip()
                if len(sentence) == 0:
                    continue
                try:
                    rewrite_results = (
                        self.test_rewrite_mode(
                            sentence=sentence,
                            task=task,
                            direction=direction,
                            num_return_sequences=num_return_sequences,
                            max_length_ratio=rewrite_max_length_ratio,
                            temperature=temperature,
                            do_sample=do_sample,
                            num_beams=num_beams
                        )
                    )
                except Exception as e:
                    print(f"[Skip rewrite] sent_id={sent_id}, error={e}")
                    continue

                # Add sent_id because test_rewrite_mode() still handles one sentence at a time.
                for result in rewrite_results:
                    result["sent_id"] = sent_id
                all_results.extend(rewrite_results)

        return all_results

    def main(self, data_path, total_num=300, fill_batch=64):
        # 1. Load dataset
        with open(data_path, "r", encoding="utf-8") as f:
            dataset_test_data = json.load(f)
        task = "cnndm" if "cnn_dm" in data_path else "yelp"
        max_level = 4 if task == "cnndm" else 5

        candidate_save_path = f"{self.cur_path}/{task}_{self.s_llm_name}_edits.json"
        if os.path.exists(candidate_save_path):
            return None

        # Candidate generation results
        # [
        #     [candidate, candidate, ...],   # sample 0
        #     [candidate, candidate, ...],   # sample 1
        #     ...
        # ]
        all_results = []

        # 2. Iterate style levels
        for style_level in range(max_level):
            print(f"\n========== style_level: {style_level} ==========")

            # 3. Iterate samples
            for sample_id, sample in enumerate(dataset_test_data[str(style_level)]):
                # Temporary debugging limit Delete this for full experiment
                if sample_id >= total_num:
                    break
                sample_start_time = time.time()
                # 3.1 Sentence segmentation
                sentences = self.split_sentences(sample, task)

                if sample_id < 10:
                    torch.cuda.reset_peak_memory_stats()
                    self.print_gpu_memory("before generation")

                # 3.2 Generate all modes for this sample
                try:
                    sample_results = (
                        self.test_sample(
                            sentences=sentences,
                            task=task,
                            style_level=style_level,
                            num_return_sequences=3,
                            temperature=0.95,
                            do_sample=True,
                            # Fill / Replace: sample-level batching
                            fill_batch_size=fill_batch
                        )
                    )

                    # 3.3 Add dataset-level identity
                    for result in sample_results:
                        result["style_level"] = style_level
                        result["sample_id"] = sample_id
                except Exception as e:
                    print("****** sample error ******")
                    print(f"style_level={style_level}, sample_id={sample_id}")
                    print(e)
                    print("-------- error --------")
                    sample_results = []

                if sample_id < 10:
                    self.print_gpu_memory("after generation")

                # One list corresponds to one source sample
                all_results.append(sample_results)

                # 4. Timing
                sample_end_time = time.time()
                print(
                    f"sample_id:{sample_id}, "
                    f"sentences_len:{len(sentences)}, "
                    f"candidate_records:{len(sample_results)}, "
                    f"spend_time:{(sample_end_time - sample_start_time) / 60:.2f} min"
                )

                # 5. Incremental saving
                # Save after every sample to avoid losing long
                # LLM-generation experiments after interruption.
                with open(candidate_save_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
        return None

    def source_main(self, data_path):
        # 1. Load dataset
        with open(data_path, "r", encoding="utf-8") as f:
            dataset_test_data = json.load(f)
        task = "cnndm" if "cnn_dm" in data_path else "yelp"
        max_level = 4 if task == "cnndm" else 5

        save_path = f"{self.cur_path}/{task}_source_sentence.json"
        if os.path.exists(save_path):
            return None

        # Immutable source sample file
        source_samples = []

        # 2. Iterate style levels
        for style_level in range(max_level):
            print(f"\n========== style_level: {style_level} ==========")

            # 3. Iterate samples
            for sample_id, sample in enumerate(dataset_test_data[str(style_level)]):
                # Temporary debugging limit Delete this for full experiment
                if sample_id >= 300:
                    break
                sample_start_time = time.time()
                # 3.1 Sentence segmentation
                sentences = self.split_sentences(sample, task)

                # 3.2 Save immutable source structure
                source_sample_ele = {
                    "style_level": style_level,
                    "sample_id": sample_id,
                    "sentences": [
                        {
                            "sent_id": sent_id,
                            "source_sentence": sent
                        }
                        for sent_id, sent
                        in enumerate(sentences)
                    ]
                }

                source_samples.append(source_sample_ele)

            json.dump(source_samples, open(save_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return None

if __name__ == "__main__":
    root_path = r"F:\python_code\StyleTrans2"
    llm_root_path = r"F:\python_code\pre_model"
    # root_path = r"/root/private_data/StyleTrans2"
    # llm_root_path = r"/root/private_data/"
    llm_name = "gemma-4-E2B-it"
    llm_dict = {"gemma-4-12B-it": "ge12B",
                "gemma-4-E2B-it": "geE2B",
                "Phi-4-mini-instruct": "phi4mini",
                "Phi-4": "phi4"}

    cg_class = CandidateOut(root_path, llm_root_path, llm_name, llm_dict)

    data_file_path = f"{root_path}/data/cnn_dm_test.json"
    cg_class.source_main(data_file_path)
    # cg_class.main(data_file_path, total_num=300, fill_batch=16)
























