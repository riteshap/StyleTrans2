import itertools
import json
import os
import pickle

from Evaluator import TextEvaluator


class SentenceCandidate:
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

    # 1. Load evaluator output
    def load_candidates(self, json_path):
        """
        Load evaluator output.
        Returns
        -------
        local_candidate_dict : dict
            Fill / Replace atomic edit candidates.

            {
                (style_level, sample_id, sent_id, direction, mode): [
                    candidate1,
                    candidate2,
                    ...
                ]
            }
        rewrite_candidate_dict : dict
            Rewrite sentence-level candidates.

            {
                (style_level, sample_id, sent_id, direction, "rewrite"): [
                    candidate1,
                    candidate2,
                    ...
                ]
            }
        """

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        local_candidate_dict = {}
        rewrite_candidate_dict = {}

        for sample_ele in data:
            for candidate in sample_ele:
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

        return local_candidate_dict, rewrite_candidate_dict

    def generate_edit_combinations(self, edits, direction, max_combination=5, max_candidates=100):
        """
        Generate non-overlapping edit combinations and uniformly
        select at most max_candidates combinations according to
        the summed atomic delta_style proxy.

        Two-pass implementation:

        Pass 1:
            Only calculate the proxy score range.
            No combinations are stored.

        Pass 2:
            Re-enumerate combinations and retain only the
            combination closest to each uniformly distributed
            target proxy score.

        This avoids storing all combinations in memory.
        """

        total_edits = len(edits)
        if total_edits == 0:
            return []

        max_n = min(max_combination, total_edits)
        # Helper: calculate proxy movement
        def calculate_proxy_delta(combination):
            raw_delta = sum(edit["delta_style"] for edit in combination)

            if direction in ["positive", "simpler"]:
                return raw_delta
            elif direction in ["negative", "complex"]:
                return -raw_delta
            else:
                raise ValueError(f"Unknown direction: {direction}")

        # Pass 1: Find score range only
        min_delta = float("inf")
        max_delta = float("-inf")
        valid_count = 0
        for n in range(1, max_n + 1):
            for combination in itertools.combinations(edits, n):
                if not self.check_overlap(combination):
                    continue
                proxy_delta = calculate_proxy_delta(combination)
                if proxy_delta < min_delta:
                    min_delta = proxy_delta
                if proxy_delta > max_delta:
                    max_delta = proxy_delta
                valid_count += 1

        # No valid combinations
        if valid_count == 0:
            return []

        # Special case:
        # fewer than max_candidates combinations
        # Re-enumerate and directly collect all.
        if valid_count <= max_candidates:
            selected = []
            for n in range(1, max_n + 1):
                for combination in itertools.combinations(edits, n):
                    if not self.check_overlap(combination):
                        continue
                    proxy_delta = (calculate_proxy_delta(combination))
                    selected.append((proxy_delta, list(combination)))
            selected.sort(key=lambda x: x[0])
            return [combination for _, combination in selected]

        # Special case: all proxy scores are identical
        if max_delta == min_delta:
            selected = []
            for n in range(1, max_n + 1):
                for combination in itertools.combinations(edits, n):
                    if not self.check_overlap(combination):
                        continue
                    selected.append(list(combination))
                    if len(selected) >= max_candidates:
                        return selected

            return selected

        # Build uniformly distributed target positions
        step = (max_delta - min_delta) / (max_candidates - 1)

        target_deltas = [min_delta + i * step for i in range(max_candidates)]

        # Pass 2:
        # Keep only the closest candidate for each target.
        # selected[target_id] =
        #     {
        #         "distance": ...,
        #         "proxy_delta": ...,
        #         "combination": ...
        #     }
        selected = [None for _ in range(max_candidates)]
        for n in range( 1, max_n + 1):
            for combination in itertools.combinations(edits, n):
                if not self.check_overlap(combination):
                    continue
                proxy_delta = calculate_proxy_delta(combination)

                # Find nearest target position directly.
                # No need to compare against all 100 targets.
                target_index = round((proxy_delta - min_delta) / step)
                target_index = max(0, min(max_candidates - 1, target_index))
                target_delta = (target_deltas[target_index])
                distance = abs(proxy_delta - target_delta)
                current = (selected[target_index])
                if current is None or distance < current["distance"]:
                    selected[target_index] = {
                        "distance": distance,
                        "proxy_delta": proxy_delta,
                        "combination": list(combination)
                    }

        # Remove empty bins
        selected = [item for item in selected if item is not None]
        # Weak -> strong
        selected.sort(key=lambda x: x["proxy_delta"])
        return [item["combination"] for item in selected]

    # 3. Check edit overlap
    def check_overlap(self, edits):
        """
        Check whether multiple edits overlap.
        Returns bool
        True: no overlap
        False: overlap exists
        """
        if len(edits) <= 1:
            return True
        sorted_edits = sorted(edits, key=lambda x:x["start_char"])
        for i in range(len(sorted_edits)-1):
            current = sorted_edits[i]
            next_edit = sorted_edits[i+1]
            if current["end_char"] > next_edit["start_char"]:
                return False
        return True

    # 4. Apply edits
    def apply_edits(self, sentence, edits):
        """
        Apply multiple local edits.
        Important:
        Replace from right to left
        to avoid character offset shift.
        Parameters sentence : str
        edits : list
        Returns str
        Modified sentence.
        """

        if not self.check_overlap(edits):
            raise ValueError("Overlapping edits cannot be applied.")
        result = sentence
        # right -> left
        sorted_edits = sorted(edits, key=lambda x:x["start_char"], reverse=True)
        for edit in sorted_edits:
            start = edit["start_char"]
            end = edit["end_char"]
            replacement = edit["generated_span"]
            result = result[:start] + replacement + result[end:]

        return result

    # 5. for one sentence candidate variants (style_level, sample_id, sent_id, direction, mode)
    def build_local_sentence_candidates(self, edits, evaluator, max_combination=5,
            semantic_threshold=0.6, filter_atomic_direction=True):
        """
        Build sentence-level candidates from atomic Fill/Replace edits.
        Parameters edits : list[dict]
            Atomic edit candidates belonging to the same:
            (sample_id, sent_id, direction, mode)
        evaluator : TextEvaluator
            Used to calculate semantic similarity and style score.
        max_combination : int
            Maximum number of edit units combined in one sentence.
        semantic_threshold : float
            Minimum sentence-level semantic similarity.
        filter_atomic_direction : bool
            Whether to remove atomic edits whose delta_style
            moves in the wrong direction before combination.
        Returns list[dict]
            Sentence-level candidates sorted according to
            the target style direction.
        """

        if len(edits) == 0:
            return []

        # 1. Basic information
        style_level = edits[0]["style_level"]
        sample_id = edits[0]["sample_id"]
        sent_id = edits[0]["sent_id"]
        direction = edits[0]["direction"]
        mode = edits[0]["mode"]
        assert mode in ["fill", "replace"]

        source_sentence = edits[0]["source_sentence"]
        source_style_score = edits[0]["source_style_score"]

        # safety check:
        # all edits must belong to the same candidate pool
        for edit in edits:
            assert edit["style_level"] == style_level
            assert edit["sample_id"] == sample_id
            assert edit["sent_id"] == sent_id
            assert edit["direction"] == direction
            assert edit["mode"] == mode
            assert edit["source_sentence"] == source_sentence

        # 2. Filter atomic edits by direction
        if filter_atomic_direction:
            if direction in ["positive", "simpler"]:  # 已确认无问题
                valid_edits = [edit for edit in edits if edit["delta_style"] > 0]
            elif direction in ["negative", "complex"]:
                valid_edits = [edit for edit in edits if edit["delta_style"] < 0]
            else:
                raise ValueError(f"Unknown direction: {direction}")
        else:
            valid_edits = edits

        if len(valid_edits) == 0:
            return []

        # 3. Generate n=1~5 edit combinations
        edit_combinations = self.generate_edit_combinations(valid_edits, direction, max_combination=max_combination)

        # 4. Apply valid combinations
        sentence_candidates = []
        # Used for sentence-level deduplication
        seen_sentences = set()
        for edit_group in edit_combinations:
            # 4.1 span conflict check
            if not self.check_overlap(edit_group):
                continue
            # 4.2 apply edits
            generated_sentence = self.apply_edits(source_sentence, edit_group)
            generated_sentence = generated_sentence.strip()
            if len(generated_sentence) == 0:
                continue

            # 4.3 sentence-level deduplication
            if generated_sentence in seen_sentences:
                continue

            seen_sentences.add(generated_sentence)

            # 4.4 semantic evaluation
            semantic_similarity = (
                evaluator.calculate_similarity(source_sentence, generated_sentence)
            )

            if semantic_similarity < semantic_threshold:
                continue

            # 4.5 style evaluation
            generated_style_score = (
                evaluator.calculate_style_score(generated_sentence)
            )

            delta_style = (generated_style_score - source_style_score)

            # 4.6 check direction after combination
            if direction in ["positive", "simpler"]:
                direction_valid = (delta_style > 0)
            elif direction in ["negative", "complex"]:
                direction_valid = (delta_style < 0)
            else:
                raise ValueError(f"Unknown direction: {direction}")

            # Combined result still moves in wrong direction
            if not direction_valid:
                continue

            # 4.7 Save candidate
            sentence_candidates.append(
                {
                    "style_level": style_level,
                    "sample_id": sample_id,
                    "sent_id": sent_id,
                    "mode": mode,
                    "direction": direction,
                    "source_sentence": source_sentence,
                    "generated_sentence": generated_sentence,
                    "source_style_score": source_style_score,
                    "generated_style_score": generated_style_score,
                    "delta_style": delta_style,
                    "semantic_similarity": semantic_similarity,
                    "edit_count": len(edit_group),
                    "edits": [
                        {
                            "start_char": edit["start_char"],
                            "end_char": edit["end_char"],
                            "original_span": edit["original_span"],
                            "generated_span": edit["generated_span"]
                        }
                        for edit in edit_group
                    ]
                }
            )

        # 5. Sort sentence candidates
        # Higher score = more positive / simpler
        if direction in ["positive", "simpler"]:
            sentence_candidates.sort(key=lambda x: x["generated_style_score"])
        # Lower score = more negative / complex
        else:
            sentence_candidates.sort(key=lambda x: x["generated_style_score"], reverse=True)

        return sentence_candidates

    def build_rewrite_sentence_candidates(self, edits, filter_atomic_direction=True):
        """
        Build sentence-level candidates from atomic Fill/Replace edits.
        Parameters edits : list[dict]
            Atomic edit candidates belonging to the same:
            (style_level, sample_id, sent_id, direction, mode)
        filter_atomic_direction : bool
            Whether to remove atomic edits whose delta_style
            moves in the wrong direction before combination.
        Returns list[dict]
            Sentence-level candidates sorted according to
            the target style direction.
        """

        if len(edits) == 0:
            return []

        # 1. Basic information
        style_level = edits[0]["style_level"]
        sample_id = edits[0]["sample_id"]
        sent_id = edits[0]["sent_id"]
        direction = edits[0]["direction"]
        mode = edits[0]["mode"]
        assert mode == "rewrite"

        source_sentence = edits[0]["source_sentence"]
        source_style_score = edits[0]["source_style_score"]

        # safety check:
        # all edits must belong to the same candidate pool
        for edit in edits:
            assert edit["style_level"] == style_level
            assert edit["sample_id"] == sample_id
            assert edit["sent_id"] == sent_id
            assert edit["direction"] == direction
            assert edit["mode"] == mode
            assert edit["source_sentence"] == source_sentence

        # 2. Filter atomic edits by direction
        if filter_atomic_direction:
            if direction in ["positive", "simpler"]:  # 已确认无问题
                valid_edits = [edit for edit in edits if edit["delta_style"] > 0]
            elif direction in ["negative", "complex"]:
                valid_edits = [edit for edit in edits if edit["delta_style"] < 0]
            else:
                raise ValueError(f"Unknown direction: {direction}")
        else:
            valid_edits = edits

        if len(valid_edits) == 0:
            return []

        # 3. Apply valid combinations
        sentence_candidates = []
        for edit in valid_edits:
            sentence_candidates.append(
                {
                    "style_level": style_level,
                    "sample_id": sample_id,
                    "sent_id": sent_id,
                    "mode": mode,
                    "direction": direction,
                    "source_sentence": source_sentence,
                    "generated_sentence": edit["generated_sentence"],
                    "source_style_score": source_style_score,
                    "generated_style_score": edit["generated_style_score"],
                    "delta_style": edit["delta_style"],
                    "semantic_similarity": edit["semantic_similarity"],
                    "edit_count": None,
                    "edits": None
                }
            )
        # 5. Sort sentence candidates
        # Higher score = more positive / simpler
        if direction in ["positive", "simpler"]:
            sentence_candidates.sort(key=lambda x: x["generated_style_score"])
        # Lower score = more negative / complex
        else:
            sentence_candidates.sort(key=lambda x: x["generated_style_score"], reverse=True)

        return sentence_candidates

    def prune_sentence_candidates(self, candidates, max_candidates=20, max_style_range=None):
        """
        Prune sentence-level candidates by style-movement stratification.

        Parameters
        ----------
        candidates : list[dict]
            Sentence-level candidates generated by
            build_local_sentence_candidates() or
            build_rewrite_sentence_candidates().

        max_candidates : int
            Maximum number of candidates to retain.

            Example:
                20 -> divide the style range into 20 bins
                40 -> divide the style range into 40 bins

        max_style_range : float or None
            Maximum directed style movement considered.

            Default:
                cnndm -> 40.0 FRE points
                yelp  -> 2.0 sentiment points

        Returns
        -------
        list[dict]
            Pruned candidates, ordered from weaker
            style movement to stronger style movement.
        """

        if len(candidates) == 0:
            return []

        if max_candidates <= 0:
            raise ValueError("max_candidates must be greater than 0.")

        # 1. Determine default style range
        if max_style_range is None:
            if self.task == "cnndm":
                max_style_range = 40.0
            elif self.task == "yelp":
                max_style_range = 2.0
            else:
                raise ValueError(f"Unknown task: {self.task}")

        # 2. Basic information
        direction = candidates[0]["direction"]
        style_level = candidates[0]["style_level"]
        sample_id = candidates[0]["sample_id"]
        mode = candidates[0]["mode"]

        if direction not in ["positive", "negative", "simpler", "complex"]:
            raise ValueError(f"Unknown direction: {direction}")

        # All candidates should belong to the same sentence candidate pool.
        for candidate in candidates:
            assert candidate["direction"] == direction
            assert candidate["style_level"] == style_level
            assert candidate["sample_id"] == sample_id
            assert candidate["mode"] == mode

        # 3. If candidate number is already small, no pruning is necessary
        if len(candidates) <= max_candidates:
            return candidates

        # 4. Calculate bin width
        bin_width = max_style_range / max_candidates
        # Example:
        # Yelp: range = 2.0  max_candidates = 20  bin_width = 0.1
        # CNN/DM: range = 40  max_candidates = 20  bin_width = 2.0

        # 5. Assign candidates to bins
        # Each bin only keeps the candidate with the highest semantic similarity.

        best_candidate_per_bin = {}
        for candidate in candidates:
            delta_style = candidate["delta_style"]

            # Convert all directions into positive movement
            if direction in ["positive", "simpler"]:
                movement = delta_style
            else:
                movement = -delta_style
            # Ignore wrong-direction / unchanged candidates
            if movement <= 0:
                continue

            # Ignore candidates outside the predefined range
            if movement > max_style_range:
                continue

            # Determine bin index
            bin_index = int(movement / bin_width)

            # remove sentence candidate out of range
            if bin_index >= max_candidates:
                continue

            # Keep highest-semantic candidate in each bin
            if bin_index not in best_candidate_per_bin:
                best_candidate_per_bin[bin_index] = candidate
            else:
                old_candidate = best_candidate_per_bin[bin_index]
                if (candidate["semantic_similarity"]
                    >
                    old_candidate["semantic_similarity"]):
                    best_candidate_per_bin[bin_index] = candidate

        # 6. Restore candidate ladder
        pruned_candidates = []
        for bin_index in sorted(best_candidate_per_bin.keys()):
            candidate = best_candidate_per_bin[bin_index]
            pruned_candidates.append(candidate)

        return pruned_candidates

    def main(self, te:TextEvaluator):
        eva_result_path = f"{self.cur_path}/{task}_{self.s_llm_name}_evaluated.json"
        local_cand_dict, rewrite_cand_dict = self.load_candidates(eva_result_path)

        # sentence_candidate_results = {}
        sentence_cand_path = f"{self.cur_path}/{self.task}_{self.s_llm_name}_sentence-cand.pkl"
        # sample_key = (1, 0, "positive", "replace")
        if os.path.exists(sentence_cand_path):
            return None

        # Fill / Replace
        for cand_key, edits in local_cand_dict.items():

            print(cand_key)
            sent_cand = self.build_local_sentence_candidates(edits=edits, evaluator=te)
            prune_sent_cand = self.prune_sentence_candidates(sent_cand)
            if len(prune_sent_cand) > 0:
                print(f"len(prune_sent_cand) = {len(prune_sent_cand)}")
                with open(sentence_cand_path, "ab") as f:
                    pickle.dump((cand_key, prune_sent_cand), f, protocol=pickle.HIGHEST_PROTOCOL)

                del prune_sent_cand

        # Rewrite
        for cand_key, edits in rewrite_cand_dict.items():

            print(cand_key)
            sent_cand = self.build_rewrite_sentence_candidates(edits=edits)
            prune_sent_cand = self.prune_sentence_candidates(sent_cand)
            if len(prune_sent_cand) > 0:
                print(f"len(prune_sent_cand) = {len(prune_sent_cand)}")
                with open(sentence_cand_path, "ab") as f:
                    pickle.dump((cand_key, prune_sent_cand), f, protocol=pickle.HIGHEST_PROTOCOL)

        return None


if __name__ == "__main__":
    task = "cnndm"
    root_path = r"F:\python_code\StyleTrans2"
    llm_name = "gemma-4-E2B-it"
    llm_dict = {"gemma-4-12B-it": "ge12B",
                "gemma-4-E2B-it": "geE2B",
                "Phi-4-mini-instruct": "phi4mini",
                "Phi-4": "phi4"}

    sbert_model_path = r"F:\python_code\pre_model\all-mpnet-base-v2"
    yelp_model_path = r"F:\python_code\pre_model\distilbert-base-uncased"
    yelp_checkpoint_path = r"F:\python_code\StyleTrans2\model_pth\yelp_distilbert_best.pt"

    enable_yelp = True if task == "yelp" else False
    te = TextEvaluator(task, root_path, llm_name, llm_dict, sbert_model_path, enable_yelp=enable_yelp,
                       yelp_model_path=yelp_model_path, yelp_checkpoint_path=yelp_checkpoint_path)

    stc = SentenceCandidate(task, root_path, llm_name, llm_dict)
    stc.main(te)