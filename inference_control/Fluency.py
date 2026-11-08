
import os
import torch
import json

from transformers import AutoTokenizer, AutoModelForCausalLM



class FluencyEvaluator:

    def __init__(self, task, root_path, llm_name, llm_dict, model_path, max_length=None):
        """
        Fluency evaluator based on Llama-3.2-3B.

        Parameters
        ----------
        model_path : str
            Local path of Llama-3.2-3B.

        max_length : int or None
            Maximum input token length.
            If None, no manual truncation is applied.
        """
        assert task in ["cnndm", "yelp"]
        self.task = task
        self.root_path = root_path
        self.cur_path = self.root_path + "/inference_control"
        self.llm_name = llm_name
        self.llm_dict = llm_dict
        self.s_llm_name = self.llm_dict[llm_name]

        self.model_path = model_path
        self.max_length = max_length
        self.tokenizer, self.model = self._load_model()

    # 1. Load evaluator model
    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            use_fast=True
        )

        # Llama normally does not define a pad token.
        # This is mainly for possible later batch evaluation.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = (
                tokenizer.eos_token
            )

        model = (
            AutoModelForCausalLM
            .from_pretrained(
                self.model_path,
                dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True
            )
        )

        model.eval()

        return tokenizer, model

    # 2. Calculate sample-level perplexity
    def _calculate_ppl(self, sample):
        """
        Calculate perplexity for one complete sample.

        Parameters
        ----------
        sample : str
            Complete source or transferred sample.

        Returns
        -------
        dict
            {
                "ppl": ...,
                "mean_nll": ...,
                "nll_sum": ...,
                "token_count": ...
            }
        """

        if not isinstance(sample, str):
            raise TypeError("sample must be a string.")
        sample = sample.strip()
        if len(sample) == 0:
            raise ValueError("sample cannot be empty.")

        # 2.1 Tokenization
        tokenize_kwargs = {"return_tensors": "pt"}
        if self.max_length is not None:
            tokenize_kwargs["truncation"] = True
            tokenize_kwargs["max_length"] = self.max_length

        inputs = self.tokenizer(sample, **tokenize_kwargs)

        # Put input on the device holding
        # the model input embeddings.
        input_device = self.model.get_input_embeddings().weight.device
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)

        # 2.2 Causal-LM loss
        # labels=input_ids lets Hugging Face internally
        # perform the standard next-token shift:
        # x_1 ... x_{n-1}
        #       ↓
        # x_2 ... x_n
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

        mean_nll = outputs.loss.float().item()
        # First token has no previous token to predict from.
        token_count = int(attention_mask.sum().item()) - 1

        if token_count <= 0:
            raise ValueError("The sample is too short to calculate perplexity.")
        nll_sum = mean_nll * token_count

        ppl = float(torch.exp(torch.tensor(mean_nll)).item())

        return {
            "ppl": ppl,
            "mean_nll": mean_nll,
            "nll_sum": nll_sum,
            "token_count": token_count
        }

    def sample_rewrite_main(self, mode="base"):

        assert mode in ["base", "sft"]
        save_path = self.cur_path + f"/{self.task}_{self.s_llm_name}_{mode}_sample_rewrite_results.json"
        if os.path.exists(save_path):
            print(f"sample_rewrite_main(), file already exists, skip")
            return None

        file_path = self.cur_path + f"/{self.task}_{self.s_llm_name}_{mode}_sample_rewrite_v2.json"
        data_file = json.load(open(file_path, "r", encoding="utf-8"))

        for dir_list in data_file:
            for ele in dir_list:
                if ele["sample_id"] % 10 == 0:
                    print(f"style_level:{ele["style_level"]}; sample_id:{ele['sample_id']}")
                source_text = ele["source_sample"]
                source_fluency_dict = self._calculate_ppl(source_text)
                ele["source_ppl"] = source_fluency_dict["ppl"]
                ele["source_token_count"] = source_fluency_dict["token_count"]

                generated_ppl_list = []
                generated_token_count_list = []
                raw_output_list = ele["raw_output"]
                for gen_text in raw_output_list:
                    fluency_dict = self._calculate_ppl(gen_text)
                    generated_ppl_list.append(fluency_dict["ppl"])
                    generated_token_count_list.append(fluency_dict["token_count"])

                ele["generated_ppl"] = generated_ppl_list
                ele["generated_token_count"] = generated_token_count_list

        json.dump(data_file, open(save_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        return None

    def main(self):
        save_path = self.cur_path + f"/{self.task}_{self.s_llm_name}_results.json"
        if os.path.exists(save_path):
            print(f"main(), file already exists, skip")
            return None

        combination_data_name = f"{self.task}_{self.s_llm_name}_combination.json"
        combination_data = json.load(open(os.path.join(self.cur_path, combination_data_name), "r", encoding="utf-8"))
        for ind, ele in enumerate(combination_data):
            ele_key = ele["key"]  # style_level, sample_id, direction, mode
            if ele_key[1] % 10 == 0:
                print(f"style_level, sample_id, direction, mode: {ele_key}")
            results = ele["results"]
            source_text = results[0]["source_text"]
            # source sample fluency
            source_fluency_dict = self._calculate_ppl(source_text)
            for ele2 in results:
                ele2["source_ppl"] = source_fluency_dict["ppl"]
                ele2["source_token_count"] = source_fluency_dict ["token_count"]
                generated_text = ele2["generated_text"]
                fluency_dict = self._calculate_ppl(generated_text)
                ele2["generated_ppl"] = fluency_dict["ppl"]
                ele2["generated_token_count"] = fluency_dict["token_count"]

        json.dump(combination_data, open(save_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        return None


if __name__ == "__main__":
    task = "yelp"
    mode = "sft"
    root_path = r"F:\python_code\StyleTrans2"
    llm_name = "gemma-4-12B-it"
    llm_dict = {"gemma-4-12B-it": "ge12B",
                "gemma-4-E2B-it": "geE2B",
                "Phi-4-mini-instruct": "phi4mini",
                "Phi-4": "phi4"}
    model_path = r"F:\python_code\pre_model\Llama-3.2-3B-Instruct"

    fe = FluencyEvaluator(task, root_path, llm_name, llm_dict, model_path)
    fe.main()
    if mode == "base":
        fe.sample_rewrite_main()
    else:
        fe.sample_rewrite_main(mode="sft")

    llm_name = "gemma-4-E2B-it"
    fe = FluencyEvaluator(task, root_path, llm_name, llm_dict, model_path)
    fe.main()
    if mode == "base":
        fe.sample_rewrite_main()
    else:
        fe.sample_rewrite_main(mode="sft")