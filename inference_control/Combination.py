
import itertools
import json
import os
import pickle

from Evaluator import TextEvaluator


class Combination:
    def __init__(self, task, root_path, llm_name, llm_dict):
        """
        Parameters  task : str  yelp or cnndm
        """
        assert task in ["yelp", "cnndm"]
        self.task = task
        self.root_path = root_path
        self.cur_path = self.root_path + "/inference_control"
        self.llm_name = llm_name
        self.llm_dict = llm_dict
        self.s_llm_name = self.llm_dict[llm_name]

    def load_sentence_candidate_results(self, save_path):
        sentence_candidate_results = {}
        with open(save_path, "rb") as f:
            while True:
                try:
                    cand_key, prune_sent_cand = (pickle.load(f))
                except EOFError:
                    break
                sentence_candidate_results[cand_key] = prune_sent_cand

        return sentence_candidate_results

    def organize_sentence_candidates(self, sentence_candidate_results):
        """
        Convert sentence-level candidate pools:
        (style_level, sample_id, sent_id, direction, mode)
        into sample-level groups:
        (style_level, sample_id, direction, mode)
            -> {
                sent_id: [sentence candidates]
            }
        """
        sample_candidate_dict = {}
        for key, candidates in sentence_candidate_results.items():
            style_level, sample_id, sent_id, direction, mode = key
            sample_key = (style_level, sample_id, direction, mode)

            if sample_key not in sample_candidate_dict:
                sample_candidate_dict[sample_key] = {}

            sample_candidate_dict[sample_key][sent_id] = candidates

        return sample_candidate_dict

    def load_source_samples(self, source_json_path):
        """
        Load source_sample.json and organize source sentences by:
            (style_level, sample_id)
                -> {sent_id: source_sentence}
        Returns: dict
        Example:
        {
            (0, 0): {
                0: "It was my first day at job.",
                1: "I have to reach at my office at 9:00 am.",
                2: "I called Orange taxi ..."
            }
        }
        """

        with open(source_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        source_sample_dict = {}
        for sample_ele in data:
            style_level = sample_ele["style_level"]
            sample_id = sample_ele["sample_id"]
            sample_key = (style_level, sample_id)
            sentence_dict = {}
            for sent_ele in sample_ele["sentences"]:
                sent_id = sent_ele["sent_id"]
                source_sentence = sent_ele["source_sentence"]
                sentence_dict[sent_id] = source_sentence
            source_sample_dict[sample_key] = sentence_dict

        return source_sample_dict

    def decide_target_delta(self, source_style_score, direction):

        if self.task == "yelp":
            source_label = int(source_style_score + 0.5)
            source_label = max(0, min(4, source_label))

            if direction == "positive":
                target_label = min(4, source_label + 1)
            elif direction == "negative":
                target_label = max(0, source_label - 1)
            else:
                raise ValueError(f"Unknown direction: {direction}")

            return abs(target_label - source_style_score)

        elif self.task == "cnndm":
            fre_ranges = [(0, 40), (40, 60), (60, 80), (80, 100)]
            source_level = None

            for i, (lower, upper) in enumerate(fre_ranges):
                if lower <= source_style_score < upper:
                    source_level = i
                    break

            if source_level is None:
                raise ValueError(f"FRE score out of range: {source_style_score}")

            if direction == "simpler":
                target_level = source_level + 1
            elif direction == "complex":
                target_level = source_level - 1
            else:
                raise ValueError(f"Unknown direction: {direction}")

            if target_level < 0 or target_level >= len(fre_ranges):
                raise ValueError(f"No target level for source_level={source_level}, direction={direction}")

            target_lower, target_upper = fre_ranges[target_level]
            target_score = (target_lower + target_upper) / 2.0

            return abs(target_score - source_style_score)

        else:
            raise ValueError(f"Unknown task: {self.task}")

    def coarse_filter_sentence_candidates(self, sample_key, sentence_candidates, source_sample_dict,
            evaluator, n_levels=21, joiner=" "):
        """
        First-stage coarse filtering.
            - Source sentence is implicitly represented by None.
            - Final filtered_candidates keeps only modified candidates.
            - Therefore one sentence may have 0, 1, or 2 candidates.
        """

        style_level, sample_id, direction, mode = sample_key
        source_key = (style_level, sample_id)

        source_sentence_dict = source_sample_dict[source_key]
        sent_ids = sorted(source_sentence_dict.keys())

        source_text = joiner.join(source_sentence_dict[sent_id] for sent_id in sent_ids)

        source_style_score = evaluator.calculate_style_score(source_text)

        target_delta = self.decide_target_delta(source_style_score, direction)

        # 1. Build ranked option pool for each sentence None = original sentence
        option_pools = {}
        for sent_id in sent_ids:
            candidates = sentence_candidates.get(sent_id, [])

            # Assume candidate["delta_style"] is sentence-level delta.
            if direction in ["positive", "simpler"]:
                candidates = sorted(candidates, key=lambda x: x["delta_style"])
            elif direction in ["negative", "complex"]:
                candidates = sorted(candidates, key=lambda x: -x["delta_style"])
            else:
                raise ValueError(f"Unknown direction: {direction}")

            option_pools[sent_id] = [None] + candidates

        # 2. Map each sentence pool to n relative levels
        level_options = {}
        for sent_id in sent_ids:
            pool = option_pools[sent_id]
            m = len(pool)
            if m == 1:
                indices = [0] * n_levels
            else:
                indices = [
                    round(i * (m - 1) / (n_levels - 1))
                    for i in range(n_levels)
                ]

            level_options[sent_id] = [pool[idx] for idx in indices]

        # 3. Construct n complete samples and calculate delta
        coarse_deltas = []
        for level in range(n_levels):
            generated_sentences = []
            for sent_id in sent_ids:
                option = level_options[sent_id][level]
                if option is None:
                    sentence = source_sentence_dict[sent_id]
                else:
                    sentence = option["generated_sentence"]
                generated_sentences.append(sentence)
            generated_text = joiner.join(generated_sentences)
            generated_style_score = evaluator.calculate_style_score(generated_text)

            raw_delta = generated_style_score - source_style_score

            # Convert both directions into positive movement.
            if direction in ["positive", "simpler"]:
                delta = raw_delta
            else:
                delta = -raw_delta

            coarse_deltas.append(delta)

        # 4. Find the adjacent pair jointly closest to target
        # Example: 0.83, 1.06, 0.95  choose 1.06 / 0.95
        best_level = min(range(n_levels - 1), key=lambda i: abs(coarse_deltas[i] - target_delta)
                                                            + abs(coarse_deltas[i + 1] - target_delta))
        level_a = best_level
        level_b = best_level + 1

        # 5. Reduce each sentence to candidates from these two adjacent levels
        level_a_candidates = {}
        level_b_candidates = {}

        for sent_id in sent_ids:
            option_a = level_options[sent_id][level_a]
            option_b = level_options[sent_id][level_b]

            # None means using the original source sentence.
            level_a_candidates[sent_id] = option_a
            level_b_candidates[sent_id] = option_b

        coarse_info = {
            "deltas": coarse_deltas,
            "target_delta": target_delta,
            "selected_levels": (level_a, level_b),
            "selected_deltas": (coarse_deltas[level_a], coarse_deltas[level_b])
        }

        return level_a_candidates, level_b_candidates, coarse_info

    def build_uniform_top_n_candidates(self, coarse_state_a, coarse_state_b, source_sample_dict,
            sample_key, evaluator, top_n=20, joiner=" "):
        """
        Build uniformly distributed Top-N sample candidates
        between two adjacent coarse states using sentence-level
        delta_style as proxy.
        """

        style_level, sample_id, direction, mode = sample_key
        source_sentence_dict = source_sample_dict[(style_level, sample_id)]
        sent_ids = sorted(source_sentence_dict.keys())

        # 1. Convert sentence delta to unified positive direction
        def directed_delta(option):
            if option is None:
                return 0.0
            delta = option["delta_style"]
            if direction in ["positive", "simpler"]:
                return delta
            elif direction in ["negative", "complex"]:
                return -delta
            else:
                raise ValueError(f"Unknown direction: {direction}")

        # 2. Find sentence positions where A and B differ
        fixed_options = {}
        switch_options = {}
        for sent_id in sent_ids:
            option_a = coarse_state_a[sent_id]
            option_b = coarse_state_b[sent_id]
            text_a = (
                source_sentence_dict[sent_id]
                if option_a is None
                else option_a["generated_sentence"]
            )
            text_b = (
                source_sentence_dict[sent_id]
                if option_b is None
                else option_b["generated_sentence"]
            )
            if text_a == text_b:
                fixed_options[sent_id] = option_a
            else:
                switch_options[sent_id] = (option_a, option_b)
        switch_ids = list(switch_options.keys())

        # 3. Enumerate all binary combinations Only calculate cheap sentence-level proxy score
        proxy_candidates = []
        for mask in range(1 << len(switch_ids[:14])):
            selected_options = dict(fixed_options)
            proxy_score = 0.0
            for i, sent_id in enumerate(switch_ids):
                option_a, option_b = switch_options[sent_id]
                if mask & (1 << i):
                    option = option_b
                else:
                    option = option_a
                selected_options[sent_id] = option

            # Total sentence-level proxy score
            for sent_id in sent_ids:
                proxy_score += directed_delta(selected_options[sent_id])

            proxy_candidates.append({
                "options": selected_options,
                "proxy_score": proxy_score
            })

        # 4. Sort according to sentence-level proxy score
        proxy_candidates.sort(key=lambda x: x["proxy_score"])

        # 5. Uniformly sample Top-N positions
        min_proxy = proxy_candidates[0]["proxy_score"]
        max_proxy = proxy_candidates[-1]["proxy_score"]

        if len(proxy_candidates) <= top_n:
            selected_candidates = proxy_candidates

        else:
            target_scores = [
                min_proxy
                + i * (max_proxy - min_proxy) / (top_n - 1)
                for i in range(top_n)
            ]

            selected_candidates = []
            used_indices = set()

            for target in target_scores:
                available_indices = [
                    i for i in range(len(proxy_candidates))
                    if i not in used_indices
                ]

                best_index = min(
                    available_indices,
                    key=lambda i: abs(proxy_candidates[i]["proxy_score"] - target)
                )

                used_indices.add(best_index)
                selected_candidates.append(proxy_candidates[best_index])

        # 6. Construct final samples and perform real
        #    sample-level evaluation only for Top-N
        source_text = joiner.join(
            source_sentence_dict[sent_id]
            for sent_id in sent_ids
        )

        source_style_score = evaluator.calculate_style_score(source_text)

        final_candidates = []
        for candidate in selected_candidates:
            generated_sentences = []
            for sent_id in sent_ids:
                option = candidate["options"][sent_id]
                if option is None:
                    sentence = source_sentence_dict[sent_id]
                else:
                    sentence = option["generated_sentence"]
                generated_sentences.append(sentence)
            generated_text = joiner.join(generated_sentences)
            generated_style_score = evaluator.calculate_style_score(generated_text)
            semantic_similarity = evaluator.calculate_similarity(
                source_text,
                generated_text
            )
            delta_style = generated_style_score - source_style_score

            final_candidates.append({
                "style_level": style_level,
                "sample_id": sample_id,
                "direction": direction,
                "mode": mode,
                "source_text": source_text,
                "generated_text": generated_text,
                "proxy_score": candidate["proxy_score"],
                "source_style_score": source_style_score,
                "generated_style_score": generated_style_score,
                "delta_style": delta_style,
                "semantic_similarity": semantic_similarity
            })

        return final_candidates

    def main(self, te: TextEvaluator):
        source_sent_path = f"{self.cur_path}/{self.task}_source_sentence.json"
        sentence_cand_path = f"{self.cur_path}/{self.task}_{self.s_llm_name}_sentence-cand.pkl"
        # sample_key = (1, 0, "positive", "replace")
        if os.path.exists(sentence_cand_path):
            sentence_candidate_results = self.load_sentence_candidate_results(sentence_cand_path)
        else:
            raise Exception("sentence_cand_path does not exist")

        save_path = f"{self.cur_path}/{self.task}_{self.s_llm_name}_combination.json"
        if os.path.exists(save_path):
            results = json.load(open(save_path, "r", encoding="utf-8"))
        else:
            results = []

        print("step2")
        sample_candidate_dict = self.organize_sentence_candidates(sentence_candidate_results)
        source_json_dict = self.load_source_samples(source_sent_path)

        print("step3")
        # sample_key = (style_level, sample_id, direction, mode)
        # build_sample_candidates, 由于采用了笛卡尔积构建候选，需要将输出结果固化到本地

        def search_sample_key(key_name, data_json):
            ret_flag = False
            for ele in data_json:
                if key_name == ele["key"]:
                    ret_flag = True
                    break
            return ret_flag

        for sample_key in sample_candidate_dict.keys():
            skip_flag = search_sample_key(sample_key, results)
            if skip_flag:
                continue

            print(sample_key)
            level_a_cand, level_b_cand, part1_info = self.coarse_filter_sentence_candidates(
                sample_key, sample_candidate_dict[sample_key], source_json_dict, te)

            sample_candidates = self.build_uniform_top_n_candidates(level_a_cand, level_b_cand,
                                                                    source_json_dict, sample_key, te)

            results.append({"key":sample_key, "results": sample_candidates})

            if sample_key[1] % 10 == 0:
                json.dump(results, open(save_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        json.dump(results, open(save_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("over")


if __name__ == "__main__":
    task = "cnndm"
    enable_yelp = True if task == "yelp" else False
    root_path = r"F:\python_code\StyleTrans2"
    llm_name = "gemma-4-E2B-it"
    llm_dict = {"gemma-4-12B-it": "ge12B",
                "gemma-4-E2B-it": "geE2B",
                "Phi-4-mini-instruct": "phi4mini",
                "Phi-4": "phi4"}

    sbert_model_path = r"F:\python_code\pre_model\all-mpnet-base-v2"
    yelp_model_path = r"F:\python_code\pre_model\distilbert-base-uncased"
    yelp_checkpoint_path = r"F:\python_code\StyleTrans2\model_pth\yelp_distilbert_best.pt"

    te = TextEvaluator(task, root_path, llm_name, llm_dict, sbert_model_path, enable_yelp=enable_yelp,
                       yelp_model_path=yelp_model_path, yelp_checkpoint_path=yelp_checkpoint_path)

    cbt = Combination(task, root_path, llm_name, llm_dict)
    cbt.main(te)

