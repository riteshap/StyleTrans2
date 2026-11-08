import os
import time
import json
import random
from openai import OpenAI


class SFTdataPrepare:
    def __init__(self, root_path, data_path, llm_name, few_shot_n=5):
        self.root_path = root_path
        self.cur_path = self.root_path + "/sft"
        self.data_path = data_path
        self.llm_name = llm_name
        self.sft_few_shot_n = few_shot_n

        self.dataset_train_data, self.task = self._load_train_data()
        pass

    def _load_train_data(self):
        # 1. Load dataset
        with open(self.data_path, "r", encoding="utf-8") as f:
            dataset_train_data = json.load(f)

        task = "cnndm" if "cnn_dm" in self.data_path else "yelp"

        return dataset_train_data, task

    def _llm_flash(self, prefix, que, api_key):
        if self.llm_name in ["deepseek-v4-flash", "deepseek-v4-pro"]:
            base_url_name = "https://api.deepseek.com"
            model_name = "deepseek-v4-flash"
        elif self.llm_name == "qwen3.5-flash":
            base_url_name = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            model_name = "qwen3.5-flash"
        elif self.llm_name == "gemini3.5-flash":
            base_url_name = "https://openrouter.ai/api/v1"
            model_name = "google/gemini-3.5-flash"
        elif self.llm_name == "nemotron3-ultra":
            base_url_name = "https://openrouter.ai/api/v1"
            model_name = "openai/gpt-5.6-luna"
        else:
            raise Exception("不存在此配置")

        client = OpenAI(api_key=api_key, base_url=base_url_name)
        for i in range(5):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prefix},
                        {"role": "user", "content": "{}".format(que)},
                    ],
                    stream=False
                )
                ss = response.choices[0].message
                if len(ss.content) == 0:
                    raise Exception("Defect2Styletext:LLM empty")
                return ss.content
            except Exception as e:
                print(e)
                print(prefix)
                print(que)
                time.sleep(3)
                if i >= 4:
                    raise Exception("times out")

    def build_sft_prompt(self, sample, task, style_level, direction):
        """
        Build few-shot prompt for SFT pseudo-parallel data generation.

        Generate three versions:
            weaker
            same
            stronger

        The three versions represent different strengths of the
        target style specified by `direction`.
        """

        task = task.lower()
        direction = direction.lower()

        if task not in ["yelp", "cnndm"]:
            raise ValueError("task must be 'yelp' or 'cnndm'.")

        # 1. Sample few-shot examples from target style level
        level_samples = (self.dataset_train_data[str(style_level)])
        candidate_examples = [text for text in level_samples if text != sample]
        few_shot_n = min(self.sft_few_shot_n, len(candidate_examples))
        few_shot_examples = random.sample(candidate_examples, few_shot_n)

        # 2. Simple prefix
        prefix = (
            "Rewrite the input while preserving its original meaning "
            "and following the style demonstrated by the examples."
        )

        # 3. Few-shot examples
        example_text = []
        for example_id, example in enumerate(few_shot_examples, start=1):
            example_text.append(f"Example {example_id}:\n{example}")
        example_text = "\n\n".join(example_text)

        # 4. Task- and direction-specific style requirement
        if task == "yelp":
            if direction == "positive":
                style_requirement = (
                    "The target style is positive sentiment. "
                    "Using the reference examples as the target sentiment strength, "
                    "rewrite the input into three versions: "
                    "the weaker version should be slightly less positive than the reference style, "
                    "the same version should match the positive sentiment strength of the reference style "
                    "as closely as possible, "
                    "and the stronger version should be slightly more positive than the reference style."
                )
            elif direction == "negative":
                style_requirement = (
                    "The target style is negative sentiment. "
                    "Using the reference examples as the target sentiment strength, "
                    "rewrite the input into three versions: "
                    "the weaker version should be slightly less negative than the reference style, "
                    "the same version should match the negative sentiment strength of the reference style "
                    "as closely as possible, "
                    "and the stronger version should be slightly more negative than the reference style."
                )
            else:
                raise ValueError("For yelp, direction must be 'positive' or 'negative'.")
        else:  # cnndm
            if direction == "simpler":
                style_requirement = (
                    "The target style is higher readability, corresponding to a higher "
                    "Flesch Reading Ease (FRE) score. "
                    "This style should use relatively simpler and shorter words, "
                    "shorter expressions, and simpler or shorter sentence structures. "
                    "Using the reference examples as the target readability strength, "
                    "rewrite the input into three versions: "
                    "the weaker version should be slightly less simple than the reference style, "
                    "the same version should match the readability of the reference style "
                    "as closely as possible, "
                    "and the stronger version should be slightly simpler and easier to read "
                    "than the reference style."
                )
            elif direction == "complex":
                style_requirement = (
                    "The target style is lower readability, corresponding to a lower "
                    "Flesch Reading Ease (FRE) score. "
                    "This style should use relatively longer or more complex words, "
                    "longer expressions, and more complex or longer sentence structures. "
                    "Using the reference examples as the target readability strength, "
                    "rewrite the input into three versions: "
                    "the weaker version should be slightly less complex than the reference style, "
                    "the same version should match the readability of the reference style "
                    "as closely as possible, "
                    "and the stronger version should be slightly more complex and harder to read "
                    "than the reference style."
                )
            else:
                raise ValueError("For cnndm, direction must be 'simpler' or 'complex'.")

        # 5. Shared generation constraints
        common_requirement = (
            "Preserve the original meaning, facts, and main information in all three versions. "
            "Keep each rewritten version approximately the same overall length as the input, "
            "and do not substantially expand, shorten, add, or remove information."
        )

        # 6. Build query
        que = (
            f"Reference examples:\n\n"
            f"{example_text}\n\n"
            f"{style_requirement}\n\n"
            f"{common_requirement}\n\n"
            f"Input:\n"
            f"{sample}\n\n"
            f"Return only the following JSON object:\n"
            f'{{\n'
            f'  "weaker": "...",\n'
            f'  "same": "...",\n'
            f'  "stronger": "..."\n'
            f'}}'
        )

        return prefix, que

    def main(self, api_key, min_sample_index=-1, max_sample_index=2000):
        """
        Generate raw pseudo-parallel data for SFT.

        This function only:
            1. iterates source samples;
            2. determines valid transfer directions;
            3. builds prompts;
            4. sends LLM requests;
            5. saves the raw LLM responses.

        No parsing, format validation, style validation,
        or semantic validation is performed here.
        """
        task = self.task
        max_level = 4  if task == "cnndm"  else 5

        # 1. Save path
        save_path = f"{self.cur_path}/{task}_sft-traindata_{min_sample_index}_{max_sample_index}.json"
        if os.path.exists(save_path):
            print(f"SFT train data already exists: {save_path}")
            all_results = json.load(open(save_path, "r", encoding="utf-8"))
        else:
            all_results = []
        os.makedirs(self.cur_path, exist_ok=True)

        def search_record(all_results, source_style_level, target_style_level, sample_id, direction):
            flag = False
            for ele in all_results:
                if (ele["sample_id"] == sample_id and
                        ele["direction"] == direction and
                        ele["source_style_level"] == source_style_level  and
                        ele["target_style_level"] == target_style_level):
                    flag = True
                    break
            return flag

        # 2. Iterate source style levels
        for style_level in range(max_level):
            print(f"\n========== style_level: {style_level} ==========")

            # 3. Determine valid directions
            if task == "yelp":
                directions = ["positive", "negative"]
                if style_level == 0:
                    directions = ["positive"]
                elif style_level == 4:
                    directions = ["negative"]
            elif task == "cnndm":
                directions = ["simpler", "complex"]
                if style_level == 0:
                    directions = ["simpler"]
                elif style_level == 3:
                    directions = ["complex"]
            else:
                raise ValueError("task must be 'yelp' or 'cnndm'.")

            # 4. Iterate samples
            for sample_id, sample in enumerate(self.dataset_train_data[str(style_level)]):

                if not (min_sample_index < sample_id <= max_sample_index):
                    continue

                sample_start_time = time.time()
                # 5. Request each valid direction
                for direction in directions:
                    # Target adjacent level
                    if direction in ["positive", "simpler"]:
                        target_style_level = style_level + 1
                    else:
                        target_style_level = style_level - 1

                    record_flag = search_record(all_results, style_level, target_style_level, sample_id, direction)
                    if record_flag:
                        continue

                    try:
                        # 5.1 Build prompt
                        prefix, que = (
                            self.build_sft_prompt(
                                sample=sample,
                                task=task,
                                style_level=style_level,
                                direction=direction
                            )
                        )

                        # 5.2 Request LLM
                        # Keep the returned content exactly as provided by the LLM.
                        raw_output = self._llm_flash(prefix=prefix, que=que, api_key=api_key)

                        # 5.3 Save raw record
                        result = {
                            "task": task,
                            "source_style_level": style_level,
                            "target_style_level": target_style_level,
                            "sample_id": sample_id,
                            "direction": direction,
                            "source_sample": sample,
                            "prefix": prefix,
                            "que": que,
                            "raw_output": raw_output
                        }
                        all_results.append(result)
                    except Exception as e:
                        print("****** SFT request error ******")
                        print(
                            f"style_level={style_level}, "
                            f"sample_id={sample_id}, "
                            f"direction={direction}"
                        )
                        print(e)
                        print("------------ error ------------")

                        continue

                # 6. Progress
                sample_end_time = time.time()
                print(
                    f"style_level:{style_level}, "
                    f"sample_id:{sample_id}, "
                    f"directions:{len(directions)}, "
                    f"spend_time:"
                    f"{sample_end_time - sample_start_time:.2f}s"
                )

                # 7. Incremental saving
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)

        return None


if __name__ == "__main__":
    min_index = -1
    max_index = 250
    ds_api_key = "sk-"
    root_path = r"F:\python_code\StyleTrans2"
    data_path = f"{root_path}/data/yelp_train.json"
    llm_name = "deepseek-v4-flash"
    few_shot_n = 5

    sft_class = SFTdataPrepare(root_path, data_path, llm_name, few_shot_n)

    sft_class.main(ds_api_key, min_sample_index=min_index, max_sample_index=max_index)