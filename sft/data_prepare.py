


import json
import re
import os
import tempfile
import textstat
import json
import math
from pathlib import Path
from inference_control.prompt import Prompt
from inference_control.Evaluator import TextEvaluator

# parsing LLM output
class SFTDataProcessor:

    def __init__(self, input_path, output_path, keep_raw_output=True):
        """
        Initialize the SFT data processor.

        Args:
            input_path (str): Path to the original JSON file.
            output_path (str): Path to the formatted JSON file.
            keep_raw_output (bool): Whether to preserve raw_output.
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.keep_raw_output = keep_raw_output

        self.required_keys = {"weaker", "same", "stronger"}

        self.metadata_keys = ["task", "sample_id", "source_style_level",
                              "target_style_level", "direction"]

        if not self.input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        if self.input_path.suffix.lower() != ".json":
            raise ValueError("Input file must be a JSON file.")

        if self.output_path.suffix.lower() != ".json":
            raise ValueError("Output file must be a JSON file.")

        if self.input_path.resolve() == self.output_path.resolve():
            raise ValueError("Input and output paths must be different.")

    def _load_data(self):
        """Load the original JSON array."""
        with open(self.input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("The input JSON must contain a list of records.")

        if not data:
            raise ValueError("The input JSON array is empty.")

        return data

    @staticmethod
    def _reject_duplicate_keys(pairs):
        """Reject duplicate JSON keys."""
        result = {}

        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key detected: {key}")
            result[key] = value

        return result

    def _parse_raw_output(self, raw_output):
        """Parse raw_output and extract three string fields."""

        try:
            parsed = json.loads(raw_output)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"JSON parsing failed: {e}") from e

        if not isinstance(parsed, dict):
            raise ValueError("Parsed output must be a dictionary.")

        for key in ("weaker", "same", "stronger"):
            if key not in parsed or not isinstance(parsed[key], str):
                raise ValueError(f"Missing or invalid string field: {key}")

        return {
            "weak": parsed["weaker"],
            "same": parsed["same"],
            "stronger": parsed["stronger"]
        }

    def _format_record(self, record):
        """Format one SFT record."""
        if not isinstance(record, dict):
            raise ValueError("Record must be a dictionary.")

        if "raw_output" not in record:
            raise ValueError("Missing raw_output field.")

        parsed_output = self._parse_raw_output(record["raw_output"])
        formatted = record.copy()
        formatted.update(parsed_output)
        if not self.keep_raw_output:
            formatted.pop("raw_output", None)

        return formatted

    def _print_error(self, index, record, error):
        """
        Print error information without exposing
        source text or generated text.
        """
        info = {"record_index": index}
        if isinstance(record, dict):
            for key in self.metadata_keys:
                if key in record:
                    value = record[key]
                    if isinstance(value, (str, int, float, bool)):
                        info[key] = value
        print(f"\n[FORMAT ERROR] {info}")
        print(f"Reason: {error}")

    def _save_data(self, data):
        """
        Atomically save the formatted JSON array.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.output_path.parent,
                suffix=".tmp",
                delete=False
            ) as f:
                temp_path = f.name
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.output_path)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def stage1_format(self):
        """
        Stage 1: Parse and validate all DeepSeek outputs.

        Rules:
            - Continue checking after individual errors.
            - Report every malformed record.
            - Do not save if any record fails.
            - Save only when all records pass.
        """
        if os.path.exists(self.output_path):
            return None

        data = self._load_data()

        formatted_data = []
        error_count = 0

        for index, record in enumerate(data):
            try:
                formatted_record = self._format_record(record)
                formatted_data.append(formatted_record)

            except (ValueError, TypeError) as e:
                error_count += 1
                self._print_error(index, record, e)

        total_count = len(data)
        success_count = len(formatted_data)

        print("\n========== Stage 1 Summary ==========")
        print(f"Total records   : {total_count}")
        print(f"Valid records   : {success_count}")
        print(f"Invalid records : {error_count}")

        if error_count > 0:
            raise RuntimeError(
                f"Stage 1 failed: {error_count} invalid records. No output file was saved."
            )

        self._save_data(formatted_data)

        print("Stage 1 completed successfully.")
        print(f"Output saved to: {self.output_path}")

        return formatted_data


class SFTDataSelector:

    def __init__(self, task, input_path, output_path, fre_fn, evaluator: TextEvaluator=None, yelp_mode=False, yelp_threshold=0.4):
        """
        Initialize the SFT training data selector.

        Args:
            input_path: Path to the formatted JSON file.
            output_path: Path to the selected JSON file.
            fre_fn: Function for calculating FRE scores.
        """
        assert task in ["yelp", "cnndm"]
        self.task = task
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.fre_fn = fre_fn
        self.yelp_threshold = yelp_threshold
        if evaluator is None and yelp_mode is True:
            raise Exception("No evaluator provided.")
        elif yelp_mode:
            self.te = evaluator
        else:
            self.te = None

        # Representative FRE values for target levels.
        self.target_fre = {0: 30.0, 1: 50.0, 2: 70.0, 3: 90.0}

    def _get_level(self, fre):
        """Convert an FRE score into a style level."""
        if fre < 40:
            return 0
        elif fre < 60:
            return 1
        elif fre < 80:
            return 2
        else:
            return 3

    def _cnndm_evaluate_candidates(self, record):
        """
        Evaluate three candidates and select the best valid one.

        Returns:
            dict or None: Selected candidate information.
        """
        target_level = record["target_style_level"]
        target_fre = self.target_fre[target_level]
        candidates = {
            "weak": record["weak"],
            "same": record["same"],
            "stronger": record["stronger"]
        }
        valid_candidates = []
        for mode, text in candidates.items():
            fre = float(self.fre_fn(text))
            if not math.isfinite(fre):
                raise ValueError(f"Invalid FRE score for candidate: {mode}")
            level = self._get_level(fre)
            # Only accept candidates within the target level.
            if level != target_level:
                continue
            error = abs(fre - target_fre)
            valid_candidates.append({"mode": mode, "text": text, "fre": fre, "error": error})

        # Reject the source sample if no candidate is valid.
        if not valid_candidates:
            return None

        # Select the candidate closest to the target reference.
        best = min(valid_candidates, key=lambda x: x["error"])

        return best

    def _yelp_evaluate_candidates(self, record):
        """Select the best Yelp candidate using classifier scores."""

        # Evaluate the original source text.
        source_score = float(self.te.calculate_style_score(record["source_sample"]))

        if not math.isfinite(source_score):
            raise ValueError("Invalid Yelp source style score.")

        # Determine the actual source level.
        source_level = int(source_score + 0.5)
        source_level = max(0, min(4, source_level))

        direction = record["direction"].lower()

        # Determine the adjacent target level.
        if direction == "positive":

            if source_level == 4:
                return None

            target_level = source_level + 1

        elif direction == "negative":

            if source_level == 0:
                return None

            target_level = source_level - 1

        else:
            raise ValueError(f"Invalid Yelp direction: {direction}")

        candidates = {
            "weak": record["weak"],
            "same": record["same"],
            "stronger": record["stronger"]
        }

        valid_candidates = []

        # Evaluate the three generated candidates.
        for mode, text in candidates.items():

            score = float(self.te.calculate_style_score(text))

            if not math.isfinite(source_score):
                raise ValueError(f"Invalid Yelp candidate score: {mode}")

            error = abs(score - target_level)

            if error <= self.yelp_threshold:
                valid_candidates.append({
                    "mode": mode,
                    "text": text,
                    "score": score,
                    "error": error
                })

        if not valid_candidates:
            return None

        # Select the candidate closest to the target level.
        best = min(valid_candidates, key=lambda x: x["error"])

        # Attach the recalculated style information.
        best["source_style_level"] = source_level
        best["target_style_level"] = target_level
        best["source_style_score"] = source_score

        return best

    def stage2_select(self):
        """
        Stage 2: Select training samples using FRE scores.
        """
        if self.output_path.exists():
            print(f"Output file already exists: {self.output_path}")
            return None

        with open(self.input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Input data must be a JSON array.")
        selected_data = []
        mode_counts = {"weak": 0, "same": 0, "stronger": 0}
        for ind, record in enumerate(data):
            if ind % 100 == 0:
                print(f"Processing {ind} / {len(data)}")

            if self.task == "cnndm":
                best = self._cnndm_evaluate_candidates(record)

            elif self.task == "yelp":
                best = self._yelp_evaluate_candidates(record)

            else:
                raise ValueError(f"Invalid task: {self.task}")

            # Skip samples without valid candidates.
            if best is None:
                continue

            # Extract task-specific style information.
            if self.task == "cnndm":
                source_style_level = record["source_style_level"]
                target_style_level = record["target_style_level"]
                selected_style_score = best["fre"]

            elif self.task == "yelp":
                source_style_level = best["source_style_level"]
                target_style_level = best["target_style_level"]
                selected_style_score = best["score"]

            else:
                raise ValueError(f"Invalid task: {self.task}")

            # Construct the selected training record.
            selected_record = {
                "task": record["task"],
                "sample_id": record["sample_id"],
                "direction": record["direction"],
                "source_style_level": source_style_level,
                "target_style_level": target_style_level,
                "source_sample": record["source_sample"],
                "target_sample": best["text"],
                "selected_mode": best["mode"],
                "selected_style_score": selected_style_score,
                "style_error": best["error"]
            }

            selected_data.append(selected_record)
            mode_counts[best["mode"]] += 1

        # Save only after all records have been processed.
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(selected_data, f, ensure_ascii=False, indent=2)

        total = len(data)
        selected = len(selected_data)

        print("\n========== Stage 2 Summary ==========")
        print(f"Total samples    : {total}")
        print(f"Selected samples : {selected}")
        print(f"Rejected samples : {total - selected}")
        print(f"Selection rate   : {selected / total:.2%}" if total else "Selection rate   : N/A")

        for mode, count in mode_counts.items():
            print(f"{mode:8s} : {count}")

        print(f"Output saved to  : {self.output_path}")

        return selected_data


if __name__ == "__main__":
    task = "yelp"
    root_path = r"F:\python_code\StyleTrans2"
    cur_path = root_path + "/sft"
    yelp_mode = True if task == "yelp" else False
    if task == "yelp":
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
    else:
        te = None

    raw_data_path = cur_path + f"/{task}_ori_traindata.json"
    formatted_save_path = cur_path + f"/{task}_formatted.json"
    processor = SFTDataProcessor(
        input_path=raw_data_path,
        output_path=formatted_save_path,
        keep_raw_output=False
    )

    formatted_data = processor.stage1_format()

    selected_save_path = cur_path + f"/{task}_selected.json"
    selector = SFTDataSelector(
        task = task,
        input_path=formatted_save_path,
        output_path=selected_save_path,
        fre_fn=textstat.flesch_reading_ease,
        evaluator=te,
        yelp_mode = yelp_mode
    )

    selected_data = selector.stage2_select()

    sft_formatted_save_path = cur_path + f"/{task}_sft_formatted.jsonl"
    prompt_class = Prompt()
    prompt_class.export_sft_data(
        input_path=selected_save_path,
        output_path=sft_formatted_save_path)