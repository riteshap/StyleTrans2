
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import os
import re
import json
import torch
import time

from transformers import AutoProcessor, AutoModelForCausalLM

from constituency import Constituency
from prompt import Prompt
from generation import Generation

from Evaluator import TextEvaluator
from SentenceCandidate import SentenceCandidate
from Combination import Combination


class SampleCandidateOut:
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

    def test_one_sample(self, sentences, direction, task, style_level:int, mode, sample_id:int, num_return_sequences=3, fill_max_new_tokens=8,
            rewrite_max_length_ratio=1.5, temperature=0.9, do_sample=True, num_beams=1, max_words=4,
            fill_batch_size=64):
        if task == "yelp":
            assert direction in ["positive", "negative"]
        elif task == "cnndm":
            assert direction in ["simpler", "complex"]
        else:
            raise ValueError("task must be 'cnndm' or 'yelp'.")

        assert mode in ["rewrite", "fill", "replace"]

        # 2. Basic input check
        if not isinstance(sentences, list):
            raise TypeError("sentences must be a list[str].")

        mode_results = []
        if mode in ["fill", "replace"]:
            include_blank_info = True if mode == "replace" else False
            mode_results = (
                self.test_fill_mode(
                    sentences=sentences,
                    task=task,
                    direction=direction,
                    include_blank_info=include_blank_info,
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
        else:
            for sent_id, sentence in enumerate(sentences):
                sentence = sentence.strip()
                try:
                    mode_results = (
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
                for result in mode_results:
                    result["sent_id"] = sent_id

        for result in mode_results:
            result["style_level"] = style_level
            result["sample_id"] = sample_id

        return mode_results


class SampleTextEvaluator:
    def __init__(self):
        pass

    def sample_main(self, edit_list, te:TextEvaluator):
        sample_out = []
        for sent_ele in edit_list:
            style_level = sent_ele["style_level"]
            sample_id = sent_ele["sample_id"]
            sent_id = sent_ele["sent_id"]
            source_sentence = sent_ele["source_sentence"]

            source_style = te.calculate_style_score(source_sentence)
            direction = sent_ele["direction"]
            mode = sent_ele["mode"]
            start_char = sent_ele.get("start_char", None)
            end_char = sent_ele.get("end_char", None)
            original_span = sent_ele.get("original_span", None)
            if mode in ["fill", "replace"]:
                generated_sentences = sent_ele["restored_sentence"]
                generated_span = sent_ele["generated_span"]
            else:
                # rewrite
                generated_span = None
                generated_sentences = sent_ele["raw_output"]

            unique_sentences = []
            unique_spans = []
            seen = set()
            for temp_ind, sent in enumerate(generated_sentences):
                if not isinstance(sent, str):
                    continue
                sent = sent.strip()
                if len(sent) == 0:
                    continue
                if sent not in seen:
                    seen.add(sent)
                    unique_sentences.append(sent)
                    if generated_span is not None:
                        unique_spans.append(generated_span[temp_ind])
                        assert generated_span[temp_ind] in sent

            # Evaluate each candidate
            valid_candidates = []
            for temp_ind, generated_sentence in enumerate(unique_sentences):
                semantic = te.calculate_similarity(source_sentence, generated_sentence)
                # semantic filter
                if semantic < te.threshold:
                    continue
                generated_style = te.calculate_style_score(generated_sentence)
                delta_style = generated_style - source_style
                unique_span = unique_spans[temp_ind] if len(unique_spans) > 0 else None

                valid_candidates.append(
                    {
                        "style_level": style_level,
                        "sample_id": sample_id,
                        "sent_id": sent_id,
                        "mode": mode,
                        "direction": direction,
                        "start_char": start_char,
                        "end_char": end_char,
                        "original_span": original_span,
                        "generated_span": unique_span,
                        "source_sentence": source_sentence,
                        "generated_sentence": generated_sentence,
                        "source_style_score": source_style,
                        "generated_style_score": generated_style,
                        "delta_style": delta_style,
                        "semantic_similarity": semantic
                    }
                )

            sample_out.extend(valid_candidates)

        return sample_out


class SampleSentenceCandidate:
    def __init__(self):
        pass

    # 1. Load evaluator output
    def _load_candidates(self, eva_edits):
            local_candidate_dict = {}
            rewrite_candidate_dict = {}

            for candidate in eva_edits:
                mode = candidate["mode"]
                key = (
                    candidate["style_level"], candidate["sample_id"], candidate["sent_id"],
                    candidate["direction"], candidate["mode"]
                )

                # Fill / Replace
                if mode in ["fill", "replace"]:
                    if key not in local_candidate_dict:
                        local_candidate_dict[key] = []
                    local_candidate_dict[key].append(candidate)

                # Rewrite
                elif mode == "rewrite":
                    if key not in rewrite_candidate_dict:
                        rewrite_candidate_dict[key] = []

                    rewrite_candidate_dict[key].append(candidate)

            assert len(local_candidate_dict) * len(rewrite_candidate_dict) == 0

            return local_candidate_dict, rewrite_candidate_dict

    def gen_sent_cand(self, eva_edits, sc: SentenceCandidate, te:TextEvaluator):
        local_cand_dict, rewrite_cand_dict = self._load_candidates(eva_edits)

        # Fill / Replace
        f_r_results = {}
        for cand_key, edits in local_cand_dict.items():
            print(cand_key)
            sent_cand = sc.build_local_sentence_candidates(edits=edits, evaluator=te)
            prune_sent_cand = sc.prune_sentence_candidates(sent_cand)
            f_r_results[cand_key] = prune_sent_cand
            if len(prune_sent_cand) > 0:
                print(f"len(prune_sent_cand) = {len(prune_sent_cand)}")

        # Rewrite
        re_results = {}
        for cand_key, edits in rewrite_cand_dict.items():
            print(cand_key)
            sent_cand = sc.build_rewrite_sentence_candidates(edits=edits)
            prune_sent_cand = sc.prune_sentence_candidates(sent_cand)
            re_results[cand_key] = prune_sent_cand
            if len(prune_sent_cand) > 0:
                print(f"len(prune_sent_cand) = {len(prune_sent_cand)}")

        return f_r_results, re_results


class SampleCombination:
    def __init__(self):
        pass

    def organize_sentence_candidates(self, sentence_candidate_results):
        sample_candidate_dict = {}
        for key, candidates in sentence_candidate_results.items():
            style_level, sample_id, sent_id, direction, mode = key
            sample_key = (style_level, sample_id, direction, mode)

            if sample_key not in sample_candidate_dict:
                sample_candidate_dict[sample_key] = {}

            sample_candidate_dict[sample_key][sent_id] = candidates

        return sample_candidate_dict

    def gen_cands(self, cur_path, task, sent_cands, cbt: Combination, te: TextEvaluator):
        source_sent_path = f"{cur_path}/{task}_source_sentence.json"
        source_json_dict = cbt.load_source_samples(source_sent_path)
        sample_candidate_dict = self.organize_sentence_candidates(sent_cands)

        def search_sample_key(key_name, data_json):
            ret_flag = False
            for ele in data_json:
                if key_name == ele["key"]:
                    ret_flag = True
                    break
            return ret_flag

        results = []
        for sample_key in sample_candidate_dict.keys():
            skip_flag = search_sample_key(sample_key, results)
            if skip_flag:
                continue

            print(sample_key)
            level_a_cand, level_b_cand, part1_info = cbt.coarse_filter_sentence_candidates(
                sample_key, sample_candidate_dict[sample_key], source_json_dict, te)

            sample_candidates = cbt.build_uniform_top_n_candidates(level_a_cand, level_b_cand,
                                                                    source_json_dict, sample_key, te)

            results.append({"key":sample_key, "results": sample_candidates})

        return results

def main():

    test_level = 2
    test_sample_num = 50
    test_mode = "rewrite"
    test_direction = "simpler"

    root_path = r"F:/python_code/StyleTrans2"
    cur_path = root_path + r"/inference_control"
    llm_root_path = r"F:\python_code\pre_model"
    root_path = r"/root/private_data/StyleTrans2"
    cur_path = root_path + r"/inference_control"
    llm_root_path = r"/root/private_data"

    data_path = root_path + r"/data/cnn_dm_test.json"

    llm_name = "gemma-4-E2B-it"
    llm_dict = {"gemma-4-12B-it": "ge12B",
                "gemma-4-E2B-it": "geE2B",
                "Phi-4-mini-instruct": "phi4mini",
                "Phi-4": "phi4"}
    s_llm_name = llm_dict[llm_name]

    task = "cnndm" if "cnn_dm" in data_path else "yelp"

    sbert_model_path = llm_root_path + r"/all-mpnet-base-v2"
    yelp_model_path = llm_root_path + r"/distilbert-base-uncased"
    yelp_checkpoint_path = root_path + r"/model_pth/yelp_distilbert_best.pt"

    enable_yelp = True if task == "yelp" else False
    te = TextEvaluator(task, root_path, llm_name, llm_dict, sbert_model_path, enable_yelp=enable_yelp,
                       yelp_model_path=yelp_model_path, yelp_checkpoint_path=yelp_checkpoint_path)
    sc = SentenceCandidate(task, root_path, llm_name, llm_dict)
    cbt = Combination(task, root_path, llm_name, llm_dict)

    ste = SampleTextEvaluator()
    ssc = SampleSentenceCandidate()
    scbt = SampleCombination()

    # 1. Load dataset
    with open(data_path, "r", encoding="utf-8") as f:
        dataset_test_data = json.load(f)

    max_level = 4 if task == "cnndm" else 5

    co_class = SampleCandidateOut(root_path, llm_root_path, llm_name, llm_dict)

    all_results = []
    # 3. Iterate samples
    time_a = time.time()
    for sample_id, sample in enumerate(dataset_test_data[str(test_level)]):
        time_b = time.time()
        # Temporary debugging limit Delete this for full experiment
        if sample_id >= test_sample_num:
            break

        sentences = co_class.split_sentences(sample, task)
        edit_result = co_class.test_one_sample(sentences, test_direction, task, test_level, test_mode, sample_id)

        eva_edits = ste.sample_main(edit_result, te)

        f_r_results, re_results = ssc.gen_sent_cand(eva_edits, sc, te)
        cand_results = f_r_results if test_mode in ["fill", "replace"] else re_results

        final_results = scbt.gen_cands(cur_path, task, cand_results, cbt, te)
        print(f"spend_time: {(time.time() - time_b)/60}min")
    print(f"task:{task}, llm_name:{llm_name}, test_mode:{test_mode}, spend_time:{(time.time() - time_a)/60}min")


if __name__ == "__main__":

    main()


























