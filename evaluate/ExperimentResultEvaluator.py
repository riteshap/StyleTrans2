
import json
import os

import numpy as np


class ExperimentResultEvaluator:

    def __init__( self,  task, root_path, s_llm_name, semantic_threshold=0.6):
        """
        Evaluate final experimental results.
        Parameters
        ----------
        task : str
            "yelp" or "cnndm"
        semantic_threshold : float
            Minimum semantic similarity required
            for a valid transferred result.
        """
        assert task in ["yelp", "cnndm"]
        self.task = task.lower()
        self.root_path = root_path
        self.s_llm_name = s_llm_name
        self.cur_path = root_path + "/evaluate"
        self.semantic_threshold = semantic_threshold

        if self.task not in ["yelp", "cnndm"]:
            raise ValueError("task must be 'yelp' or 'cnndm'.")

    # 1. Convert continuous style score to discrete level
    def _score_to_level(self, style_score):
        """
        Convert evaluator style score into style level.
        Yelp:
            sentiment score approximately 1~5
            level 0 -> score around 1
            level 1 -> score around 2
            ...
            level 4 -> score around 5
        CNN/DM:
            level 0: FRE < 40
            level 1: 40 <= FRE < 60
            level 2: 60 <= FRE < 80
            level 3: FRE >= 80
        """

        if self.task == "yelp":
            # Example:
            # 1.07 -> level 0; 1.56 -> level 1; 2.63 -> level 2
            level = int(style_score + 0.5) - 1
            level = max(0, min(4, level))
            return level

        # CNN/DM
        if style_score < 40:
            return 0
        elif style_score < 60:
            return 1
        elif style_score < 80:
            return 2
        else:
            return 3

    # 2. Determine adjacent target level
    def _get_target_level(self, source_level, direction):
        direction = direction.lower()
        if self.task == "yelp":
            if direction == "positive":
                return min(4, source_level + 1)
            elif direction == "negative":
                return max(0, source_level - 1)
            else:
                raise ValueError("For Yelp, direction must be 'positive' or 'negative'.")

        # CNN/DM
        if direction == "simpler":
            return min(3, source_level + 1)
        elif direction == "complex":
            return max(0, source_level - 1)
        else:
            raise ValueError("For CNN/DM, direction must be 'simpler' or 'complex'.")

    # 3. Select one final result for one key
    def select_best_result(self, results):
        """
        Select one final candidate from the results
        belonging to one key.

        Selection procedure:

        1. Determine target style level.
        2. Keep candidates whose generated style score
           falls into the target level.
        3. Keep candidates with semantic similarity
           >= semantic_threshold.
        4. Select the candidate with the highest
           semantic similarity.

        Returns
        -------
        dict or None
            Best candidate if available;
            otherwise None.
        """

        if not results:
            return None

        # All results under one key should share
        # source level and direction.
        source_level = self._score_to_level(results[0]["source_style_score"])
        # source_level = results[0]["style_level"]
        direction = (results[0]["direction"])
        target_level = (self._get_target_level(source_level, direction))
        valid_results = []
        for result in results:
            generated_style_score = (result["generated_style_score"])
            generated_level = self._score_to_level(generated_style_score)

            # Must successfully enter target level
            if generated_level != target_level:
                continue

            # Semantic preservation threshold
            semantic_similarity = (result["semantic_similarity"])
            if semantic_similarity < self.semantic_threshold:
                continue

            valid_results.append(result)

        # No successful candidate
        if len(valid_results) == 0:
            return None

        # Among successful candidates, maximize semantic preservation
        best_result = max(valid_results, key=lambda x: x["semantic_similarity"])

        return best_result

    def select_best_sample_rewrite_result(self, result):
        """
        Select the best candidate from one sample-rewrite record.

        Selection:
            1. generated style must reach the target level;
            2. semantic similarity >= semantic_threshold;
            3. among valid candidates, choose the one with
               the highest semantic similarity.

        Parameters
        ----------
        result : dict
            One sample-rewrite record containing multiple
            generated candidates and their corresponding
            evaluation scores.

        Returns
        -------
        dict or None
            Standardized best candidate record.
            Returns None if no valid candidate exists.
        """

        if not result:
            return None

        # 1. Basic information
        # style_level = result["style_level"]
        style_level = self._score_to_level(result["source_style_score"])
        sample_id = result["sample_id"]
        direction = result["direction"]
        source_text = result["source_sample"]
        source_style_score = result["source_style_score"]
        target_level = self._get_target_level(style_level, direction)

        # 2. Candidate arrays
        generated_texts = result.get("raw_output", [])
        style_scores = result.get("style_score", [])
        sim_scores = result.get("sim_score", [])
        generated_ppls = result.get("generated_ppl", [])
        generated_token_counts = result.get("generated_token_count", [])

        if len(generated_texts) == 0:
            return None

        # 3. Search valid candidates
        valid_candidates = []
        candidate_num = min(
            len(generated_texts),
            len(style_scores),
            len(sim_scores),
            len(generated_ppls),
            len(generated_token_counts)
        )

        for candidate_id in range(candidate_num):
            generated_text = generated_texts[candidate_id]
            generated_style_score = style_scores[candidate_id]
            semantic_similarity = sim_scores[candidate_id]
            generated_ppl = generated_ppls[candidate_id]
            generated_token_count = generated_token_counts[candidate_id]

            # 3.1 Style-level success
            generated_level = self._score_to_level(generated_style_score)
            if generated_level != target_level:
                continue

            # 3.2 Semantic threshold
            if semantic_similarity < self.semantic_threshold:
                continue

            # 3.3 Standardize candidate format
            candidate = {
                "style_level": style_level,
                "sample_id": sample_id,
                "direction": direction,
                "mode": "sample_rewrite",
                "source_text": source_text,
                "generated_text": generated_text,
                "proxy_score": None,  # Sample rewrite has no proxy score
                "source_style_score": source_style_score,
                "generated_style_score": generated_style_score,
                "delta_style": generated_style_score - source_style_score,
                "semantic_similarity": semantic_similarity,
                "source_ppl": result["source_ppl"],
                "source_token_count": result["source_token_count"],
                "generated_ppl": generated_ppl,
                "generated_token_count": generated_token_count,
                "candidate_id": candidate_id  # Useful for later analysis
            }
            valid_candidates.append(candidate)

        # 4. No successful candidate
        if len(valid_candidates) == 0:
            return None

        # 5. Select highest-semantic candidate
        best_result = max(valid_candidates, key=lambda x: x["semantic_similarity"])

        return best_result

    def select_best_sample_rewrite_result_n(self, result, candidate_n):
        """
        Select the best successful result using only
        the first candidate_n generated candidates.

        Parameters
        ----------
        result : dict
            One sample-rewrite record.

        candidate_n : int
            Only use the first candidate_n candidates.

        Returns
        -------
        dict or None
            Best successful candidate.
        """

        if not result:
            return None

        style_level = self._score_to_level(result["source_style_score"])
        sample_id = result["sample_id"]
        direction = result["direction"]

        source_text = result["source_sample"]
        source_style_score = result["source_style_score"]

        target_level = self._get_target_level(style_level, direction)

        generated_texts = result.get("raw_output", [])
        style_scores = result.get("style_score", [])
        sim_scores = result.get("sim_score", [])
        generated_ppls = result.get("generated_ppl", [])
        generated_token_counts = result.get("generated_token_count", [])

        # Only use the first N candidates
        candidate_num = min(
            candidate_n,
            len(generated_texts),
            len(style_scores),
            len(sim_scores),
            len(generated_ppls),
            len(generated_token_counts)
        )

        if candidate_num == 0:
            return None
        valid_candidates = []
        for candidate_id in range(candidate_num):
            generated_text = (generated_texts[candidate_id])
            generated_style_score = (style_scores[candidate_id])
            semantic_similarity = (sim_scores[candidate_id])
            generated_ppl = (generated_ppls[candidate_id])
            generated_token_count = (generated_token_counts[candidate_id])

            # 1. Fine-grained transfer success
            generated_level = (self._score_to_level(generated_style_score))
            if generated_level != target_level:
                continue

            # 2. Semantic threshold
            if semantic_similarity < self.semantic_threshold:
                continue

            # 3. Standardized result
            candidate = {
                "style_level": style_level,
                "sample_id": sample_id,
                "direction": direction,
                "mode": "sample_rewrite",
                "source_text": source_text,
                "generated_text": generated_text,
                "proxy_score": None,
                "source_style_score": source_style_score,
                "generated_style_score": generated_style_score,
                "delta_style": generated_style_score - source_style_score,
                "semantic_similarity": semantic_similarity,
                "source_ppl": result["source_ppl"],
                "source_token_count": result["source_token_count"],
                "generated_ppl": generated_ppl,
                "generated_token_count": generated_token_count,
                "candidate_id": candidate_id
            }
            valid_candidates.append(candidate)
        if len(valid_candidates) == 0:
            return None

        # Among successful candidates,
        # select the one with highest semantic similarity
        best_result = max(
            valid_candidates,
            key=lambda x:
            x["semantic_similarity"]
        )

        return best_result

    # sample_rewrite result for experiment
    def sample_rewrite_main(self, data_path):
        """
        Calculate main experimental metrics for sample_rewrite.

        Metrics:
            1. Fine-grained transfer success rate
            2. Mean semantic similarity
            3. Mean source PPL
            4. Mean generated PPL

        Each direction-level record is treated as one transfer key.
        """

        # 1. Load data
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 2. Statistics containers
        total_keys = 0
        success_keys = 0

        semantic_scores = []
        source_ppls = []
        generated_ppls = []

        # 3. Iterate samples
        for sample_records in data:
            # sample_records:
            # [
            #     direction_record_1,
            #     direction_record_2
            # ]
            if not sample_records:
                continue
            for result in sample_records:
                total_keys += 1
                # Select one final successful candidate
                best_result = (self.select_best_sample_rewrite_result(result))
                if best_result is None:
                    continue

                # Successful transfer
                success_keys += 1
                semantic_scores.append(best_result["semantic_similarity"])
                source_ppls.append(best_result["source_ppl"])
                generated_ppls.append(best_result["generated_ppl"])

        # 4. Calculate statistics
        success_rate = (success_keys / total_keys
                        if total_keys > 0 else 0.0)
        semantic_mean = (float(np.mean(semantic_scores))
            if semantic_scores else None)

        source_ppl_mean = (float(np.mean(source_ppls))
            if source_ppls else None)
        generated_ppl_mean = (float(np.mean(generated_ppls))
            if generated_ppls else None)

        # 5. Return same format as previous main function
        final_results = {
            "sample_rewrite": {
                "total_keys": total_keys,
                "success_keys": success_keys,
                "success_rate": success_rate,
                "semantic_mean": semantic_mean,
                "source_ppl_mean": source_ppl_mean,
                "generated_ppl_mean": generated_ppl_mean
            }
        }

        return final_results

    def sample_rewrite_n_analysis(self, data_path, max_candidates=20):
        """
        Analyze Sample Rewrite performance when the number
        of generated candidates increases from 1 to max_candidates.

        For each N:
            - only the first N generated candidates are used;
            - transfer success rate is calculated;
            - among successful candidates, the candidate with
              highest semantic similarity is selected;
            - semantic similarity and PPL are then averaged.

        Returns
        -------
        dict
            {
                1: {...},
                2: {...},
                ...
                20: {...}
            }
        """

        # 1. Load data
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        final_results = {}

        # 2. N = 1 ... 20
        for candidate_n in range(1, max_candidates + 1):

            total_keys = 0
            success_keys = 0

            semantic_scores = []
            source_ppls = []
            generated_ppls = []

            # 3. Iterate all sample/direction records
            for sample_records in data:
                if not sample_records:
                    continue
                for result in sample_records:
                    total_keys += 1
                    best_result = (
                        self.select_best_sample_rewrite_result_n(
                            result=result,
                            candidate_n=candidate_n
                        )
                    )
                    if best_result is None:
                        continue
                    success_keys += 1
                    semantic_scores.append(best_result["semantic_similarity"])
                    source_ppls.append(best_result["source_ppl"])
                    generated_ppls.append(best_result["generated_ppl"])

            # 4. Aggregate
            success_rate = (success_keys / total_keys
                if total_keys > 0 else 0.0)
            semantic_mean = (float(np.mean(semantic_scores))
                if semantic_scores else None)
            source_ppl_mean = (float(np.mean(source_ppls))
                if source_ppls else None)
            generated_ppl_mean = (float(np.mean(generated_ppls))
                if generated_ppls else None)

            final_results[candidate_n] = {
                "n": candidate_n,
                "total_keys": total_keys,
                "success_keys": success_keys,
                "success_rate": success_rate,
                "semantic_mean": semantic_mean,
                "source_ppl_mean": source_ppl_mean,
                "generated_ppl_mean": generated_ppl_mean
            }

        return final_results

    # fill/replace/sentence rewrite result for experiment
    def main(self,  data_path):
        """
        Calculate the three main experimental metrics
        for Fill / Replace / Rewrite:

        1. Fine-grained transfer success rate
        2. Semantic preservation
        3. Fluency:
            - source sample PPL
            - transferred sample PPL

        One key corresponds to one transfer task:
            (style_level, sample_id, direction, mode)

        For every key, select_best_result() returns
        at most one final candidate.
        """
        save_path = self.cur_path + f"/{self.task}_{self.s_llm_name}_results.json"
        if os.path.exists(save_path):
            print("results is already saved")

        # 1. Load result file
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 2. Initialize statistics
        modes = ["fill", "replace", "rewrite"]
        stats = {}
        for mode in modes:
            stats[mode] = {
                "total_keys": 0,
                "success_keys": 0,
                "semantic_scores": [],
                "source_ppls": [],
                "generated_ppls": []
            }

        ret_records = []
        # 3. Iterate all keys
        for record in data:
            key = record["key"]
            results = record["results"]
            # key:[style_level, sample_id, direction, mode]
            mode = key[3]
            if mode not in stats:
                continue

            # Count one transfer task
            stats[mode]["total_keys"] += 1
            # Select one final successful candidate
            best_result = self.select_best_result(results)

            ret_records.append(best_result)

            # No candidate reached target level while satisfying semantic threshold
            if best_result is None:
                continue

            # Successful transfer
            stats[mode]["success_keys"] += 1
            stats[mode]["semantic_scores"].append(best_result["semantic_similarity"])
            stats[mode]["source_ppls"].append(best_result["source_ppl"])
            stats[mode]["generated_ppls"].append(best_result["generated_ppl"])

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(ret_records, f, ensure_ascii=False, indent=4)

        # 4. Calculate final statistics
        final_results = {}
        for mode in modes:
            mode_stats = stats[mode]
            total_keys = mode_stats["total_keys"]
            success_keys = mode_stats["success_keys"]

            # Transfer success rate
            if total_keys > 0:
                success_rate = success_keys / total_keys
            else:
                success_rate = 0.0

            # Mean semantic similarity
            semantic_mean = (float(np.mean(mode_stats["semantic_scores"]))
                if mode_stats["semantic_scores"]
                else None
            )

            # Mean source PPL
            source_ppl_mean = (float(np.mean(mode_stats["source_ppls"]))
                if mode_stats["source_ppls"]
                else None
            )

            # Mean generated PPL
            generated_ppl_mean = (float(np.mean(mode_stats["generated_ppls"]))
                if mode_stats["generated_ppls"]
                else None
            )

            # Final record
            final_results[mode] = {
                "total_keys": total_keys,
                "success_keys": success_keys,
                "success_rate": success_rate,
                "semantic_mean": semantic_mean,
                "source_ppl_mean": source_ppl_mean,
                "generated_ppl_mean": generated_ppl_mean
            }

        return final_results


if __name__ == "__main__":

    task = "cnndm"
    root_path = r"F:\python_code\StyleTrans2"
    llm_name = "gemma-4-12B-it"
    llm_dict = {"gemma-4-12B-it": "ge12B",
                "gemma-4-E2B-it": "geE2B",
                "Phi-4-mini-instruct": "phi4mini",
                "Phi-4": "phi4"}
    s_llm_name = llm_dict[llm_name]
    loc_edit_path = root_path + f"/inference_control/{task}_{s_llm_name}_results.json"
    mode = "sft"
    sample_rewrite_path = root_path + f"/inference_control/{task}_{s_llm_name}_{mode}_sample_rewrite_results.json"

    ere = ExperimentResultEvaluator(task, root_path, s_llm_name)

    ret_results = ere.main(loc_edit_path)
    print(ret_results)

    ret_results = ere.sample_rewrite_main(sample_rewrite_path)
    print(ret_results)

    ret_results = ere.sample_rewrite_n_analysis(sample_rewrite_path)
    # print(ret_results)

    print("success_rate")
    for ind in ret_results:
        ind_ele = ret_results[ind]
        print(ind_ele["success_rate"], end= " ")

    print("semantic_mean")
    for ind in ret_results:
        ind_ele = ret_results[ind]
        print(ind_ele["semantic_mean"], end=" ")

    print("generated_ppl_mean")
    for ind in ret_results:
        ind_ele = ret_results[ind]
        print(ind_ele["generated_ppl_mean"], end=" ")




