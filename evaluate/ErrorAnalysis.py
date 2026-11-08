


import os
import json
import re

from transformers import AutoProcessor


class ErrorAnalysis:

    def __init__(self, processor_path, max_edit_tokens=8, anchor_search_window=32):
        """
        Parameters
        ----------
        processor_path : str
            Local LLM processor path.

        max_edit_tokens : int
            Maximum token length allowed for one reconstructed
            local edit unit.

        anchor_search_window : int
            Maximum forward search range when looking for the
            next shared token anchor.
        """
        self.VALID_PATTERN = re.compile(r"""^[A-Za-z0-9\s.,!?;:'"()\-]*$""")
        self.processor_path = processor_path
        self.max_edit_tokens = max_edit_tokens
        self.anchor_search_window = anchor_search_window
        self.processor = self._load_processor()

        # Gemma / multimodal-style AutoProcessor normally contains a tokenizer.
        if hasattr(self.processor, "tokenizer"):
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor

    # 1. Load processor
    def _load_processor(self):
        processor = (AutoProcessor.from_pretrained(self.processor_path, local_files_only=True))
        return processor

    # 2. Encode text
    def _encode(self, text):
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return token_ids

    # 3. Decode token span
    def _decode(self, token_ids):
        if len(token_ids) == 0:
            return ""
        text = (
            self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        )

        return text.strip()

    # 4. Find the next common token anchor
    def _find_next_anchor(self, source_ids, generated_ids, source_index, generated_index):
        """
        Search forward for the nearest pair of identical tokens.

        Returns
        -------
        tuple[int, int] or None

        The returned values are offsets relative to
        source_index / generated_index.
        """

        max_source_offset = min(
            self.anchor_search_window,
            len(source_ids) - source_index - 1
        )
        max_generated_offset = min(
            self.anchor_search_window,
            len(generated_ids) - generated_index - 1
        )

        best_anchor = None
        best_cost = None
        for source_offset in range(max_source_offset + 1):
            source_pos = source_index + source_offset
            for generated_offset in range(max_generated_offset + 1):
                # Current position is already known to be different.
                if source_offset == 0 and generated_offset == 0:
                    continue
                generated_pos = generated_index + generated_offset
                if source_ids[source_pos] != generated_ids[generated_pos]:
                    continue
                # Prefer the closest shared token.
                cost = source_offset + generated_offset
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_anchor = (source_offset, generated_offset)

        return best_anchor

    # 5. Reconstruct local edit units
    def extract_local_edits(self, source_text, generated_text):
        """
        Approximately reconstruct local edits through
        greedy token-level alignment.

        Returns
        -------
        list[dict]

        Example:
        [
            {
                "source_unit": "alleged crimes",
                "generated_unit": "crimes"
            },
            ...
        ]
        """
        source_ids = self._encode(source_text)
        generated_ids = self._encode(generated_text)
        source_index = 0
        generated_index = 0
        edit_units = []
        # Greedy left-to-right alignment
        while (
            source_index < len(source_ids)
            and
            generated_index < len(generated_ids)
        ):

            # Identical token -> continue
            if source_ids[source_index] == generated_ids[generated_index]:
                source_index += 1
                generated_index += 1
                continue
            # Different token: search for the next shared anchor
            anchor = (
                self._find_next_anchor(
                    source_ids=source_ids,
                    generated_ids=generated_ids,
                    source_index=source_index,
                    generated_index=generated_index
                )
            )

            # No nearby anchor: treat remaining suffix as one possible edit
            if anchor is None:
                source_span = source_ids[source_index:]
                generated_span = generated_ids[generated_index:]
                # Only retain sufficiently local edits
                if (len(source_span) <= self.max_edit_tokens and
                        len(generated_span) <= self.max_edit_tokens):

                    edit_units.append({
                        "source_unit":self._decode(source_span),
                        "generated_unit":self._decode(generated_span)
                    })

                break

            # Tokens before the anchor form one edit span
            source_offset, generated_offset = anchor
            source_span = (source_ids[source_index: source_index + source_offset])
            generated_span = generated_ids[generated_index: generated_index + generated_offset]

            # Only retain local spans <= threshold
            # If too long, discard this unit but still jump to the newly found anchor.

            if (len(source_span) <= self.max_edit_tokens
                and len(generated_span) <= self.max_edit_tokens):

                edit_units.append({
                    "source_unit": self._decode(source_span),
                    "generated_unit": self._decode(generated_span)
                })

            # Jump to the common anchor. The anchor itself is not part of the edit.
            source_index += source_offset
            generated_index += generated_offset

        # One sequence may have remaining tokens
        if (source_index >= len(source_ids) and
            generated_index < len(generated_ids)):
            generated_span = generated_ids[generated_index:]
            if len(generated_span) <= self.max_edit_tokens:

                edit_units.append({
                    "source_unit": "",
                    "generated_unit": self._decode(generated_span)
                })

        elif (generated_index >= len(generated_ids) and
            source_index < len(source_ids)):
            source_span =source_ids[source_index:]
            if len(source_span) <= self.max_edit_tokens:

                edit_units.append({
                    "source_unit": self._decode(source_span),
                    "generated_unit": ""
                })

        return edit_units

    # 6. Main
    def main1(self, data_path, save_path, modes=("fill", "replace")):
        """
        Reconstruct local edits from the final selected
        experimental results.

        Only source/generated samples and reconstructed
        local edit units are retained.
        """
        if os.path.exists(save_path):
            output_results = json.load(open(save_path, "r", encoding="utf-8"))
            return output_results

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        output_results = []
        for result in data:
            if result is None:
                continue
            # Error analysis focuses on local-edit methods.
            if result.get("mode") not in modes:
                continue
            source_text = result["source_text"]
            generated_text = result["generated_text"]
            edit_units = self.extract_local_edits(source_text=source_text, generated_text=generated_text)

            output_results.append({
                "style_level": result["style_level"],
                "sample_id": result["sample_id"],
                "direction": result["direction"],
                "mode": result["mode"],

                "source_text": source_text,
                "generated_text": generated_text,
                "edit_units": edit_units
            })

        # Save
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output_results, f, ensure_ascii=False, indent=2)
        return output_results

    def has_invalid_chars(self, text):

        if text is None:
            return False

        return self.VALID_PATTERN.fullmatch(text) is None

    def analyze_error_rate(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_units = 0

        source_error_units = 0
        generated_error_units = 0

        source_error_samples = 0
        generated_error_samples = 0

        source_error_examples = []
        generated_error_examples = []

        for sample in data:

            sample_source_error = False
            sample_generated_error = False

            for unit in sample.get("edit_units", []):

                total_units += 1

                source_unit = unit.get("source_unit", "")
                generated_unit = unit.get("generated_unit", "")

                # 原始局部单元
                if self.has_invalid_chars(source_unit):
                    source_error_units += 1
                    sample_source_error = True

                    source_error_examples.append({
                        "sample_id": sample.get("sample_id"),
                        "text": source_unit
                    })

                # LLM生成后的局部单元
                if self.has_invalid_chars(generated_unit):
                    generated_error_units += 1
                    sample_generated_error = True

                    generated_error_examples.append({
                        "sample_id": sample.get("sample_id"),
                        "source_unit": source_unit,
                        "generated_unit": generated_unit
                    })

            if sample_source_error:
                source_error_samples += 1

            if sample_generated_error:
                generated_error_samples += 1

        total_samples = len(data)

        print("========== Edit-unit level ==========")
        print(f"Total edit units: {total_units}")

        print(
            f"Source error units: {source_error_units} "
            f"({source_error_units / total_units:.4%})"
        )

        print(
            f"Generated error units: {generated_error_units} "
            f"({generated_error_units / total_units:.4%})"
        )

        print("\n========== Sample level ==========")
        print(f"Total samples: {total_samples}")

        print(
            f"Source error samples: {source_error_samples} "
            f"({source_error_samples / total_samples:.4%})"
        )

        print(
            f"Generated error samples: {generated_error_samples} "
            f"({generated_error_samples / total_samples:.4%})"
        )

        print("\n========== Generated error examples ==========")
        for item in generated_error_examples[:20]:
            print(item)

        return {
            "total_samples": total_samples,
            "total_units": total_units,

            "source_error_units": source_error_units,
            "source_error_unit_rate":
                source_error_units / total_units if total_units else 0,

            "generated_error_units": generated_error_units,
            "generated_error_unit_rate":
                generated_error_units / total_units if total_units else 0,

            "source_error_samples": source_error_samples,
            "source_error_sample_rate":
                source_error_samples / total_samples if total_samples else 0,

            "generated_error_samples": generated_error_samples,
            "generated_error_sample_rate":
                generated_error_samples / total_samples if total_samples else 0,

            "generated_error_examples": generated_error_examples
        }

    def unique_keep_order(self, items):
        """去重，同时保持原始顺序。"""
        return list(dict.fromkeys(items))

    def analyze_original_edit_errors(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_units = 0
        error_units = 0
        error_examples = []

        for group in data:
            for edit in group:

                generated = edit.get("generated_span", [])

                generated = self.unique_keep_order(generated)

                for unit in generated:
                    total_units += 1

                    if self.has_invalid_chars(unit):
                        error_units += 1

                        error_examples.append({
                            "sent_id": edit.get("sent_id"),
                            "candidate_id": edit.get("candidate_id"),
                            "original_span": edit.get("original_span"),
                            "generated_span": unit
                        })

        error_rate = (
            error_units / total_units
            if total_units > 0 else 0
        )

        print(f"Total unique generated units: {total_units}")
        print(f"Error units: {error_units}")
        print(f"Error rate: {error_rate:.4%}")

        print("\nError examples:")
        for item in error_examples[:20]:
            print(item)

        return {
            "total_units": total_units,
            "error_units": error_units,
            "error_rate": error_rate,
            "error_examples": error_examples
        }



if __name__ == "__main__":
    task = "cnndm"
    root_path = r"F:/python_code/StyleTrans2"
    cur_path = root_path + r"/evaluate"
    llm_root_path = r"F:\python_code\pre_model"
    llm_name = "gemma-4-E2B-it"
    llm_dict = {"gemma-4-12B-it": "ge12B",
                "gemma-4-E2B-it": "geE2B",
                "Phi-4-mini-instruct": "phi4mini",
                "Phi-4": "phi4"}
    edits_path = root_path + f"/inference_control/{task}_{llm_dict[llm_name]}_edits.json"

    processor_path = llm_root_path + f"/{llm_name}"
    data_path = cur_path + f"/{task}_{llm_dict[llm_name]}_results.json"
    save_path = cur_path + f"/{task}_{llm_dict[llm_name]}_results_errors.json"

    eay = ErrorAnalysis(processor_path)
    eay.main1(data_path, save_path)
    eay.analyze_error_rate(save_path)
    eay.analyze_original_edit_errors(edits_path)


