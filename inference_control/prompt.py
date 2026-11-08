class Prompt:

    def build_fill_prompt(
        self,
        candidate,
        task,
        direction,
        include_blank_info=False
    ):
        """
        Build style-guided fill / replace prompt.
        Parameters
        ----------
        candidate : dict
            Must contain:
                candidate["masked_text"]
                candidate["text"]
        task : str
            "cnndm" or "yelp"
        direction : str  CNN/DM: "simpler"  "complex"
                        Yelp:  "positive"  "negative"
        include_blank_info : bool   False -> fill mode  True  -> replace mode
        Returns  str  User prompt for instruction-tuned LLM.
        """

        task = task.lower()
        direction = direction.lower()

        # ---------------------------------------------
        # Style direction
        # ---------------------------------------------
        if task == "cnndm":

            if direction == "simpler":
                instruction = (
                    "Fill in [BLANK] to make the sentence slightly easier to read "
                    "using simpler words and shorter expressions."
                )

            elif direction == "complex":
                instruction = (
                    "Fill in [BLANK] to make the sentence slightly harder to read "
                    "using more complex words and expressions."
                )

            else:
                raise ValueError(
                    "For cnndm, direction must be 'simpler' or 'complex'."
                )

        elif task == "yelp":

            if direction == "positive":
                instruction = (
                    "Fill in [BLANK] to make the sentence slightly more positive."
                )

            elif direction == "negative":
                instruction = (
                    "Fill in [BLANK] to make the sentence slightly more negative."
                )

            else:
                raise ValueError(
                    "For yelp, direction must be 'positive' or 'negative'."
                )

        else:
            raise ValueError(
                "task must be 'cnndm' or 'yelp'."
            )

        # Replace mode
        if include_blank_info:
            prompt = (
                f'{instruction}\n\n'
                f'Original text in [BLANK]: '
                f'"{candidate["text"]}"\n\n'
                f'Sentence:\n'
                f'{candidate["masked_text"]}\n\n'
                f'Return only the replacement text.'
            )

        # Fill mode
        else:
            prompt = (
                f'{instruction}\n\n'
                f'Sentence:\n'
                f'{candidate["masked_text"]}\n\n'
                f'Return only the missing text.'
            )

        return prompt


    def build_plain_fill_prompt(
        self,
        candidate,
        include_blank_info=False
    ):
        """
        Build fill / replace prompt without style guidance.
        Used to test the basic infilling capability of the LLM.
        """

        # Plain replace
        if include_blank_info:

            prompt = (
                f'Replace [BLANK] with an appropriate expression.\n\n'
                f'Original text in [BLANK]: '
                f'"{candidate["text"]}"\n\n'
                f'Sentence:\n'
                f'{candidate["masked_text"]}\n\n'
                f'Return only the replacement text.'
            )

        # ---------------------------------------------
        # Plain fill
        # ---------------------------------------------
        else:

            prompt = (
                f'Fill in [BLANK] with an appropriate expression.\n\n'
                f'Sentence:\n'
                f'{candidate["masked_text"]}\n\n'
                f'Return only the missing text.'
            )

        return prompt


    def build_rewrite_prompt(
        self,
        sentence,
        task,
        direction
    ):
        """
        Build full-sentence rewriting prompt.
        """

        task = task.lower()
        direction = direction.lower()

        if task == "cnndm":

            if direction == "simpler":
                instruction = (
                    "Rewrite the sentence to make it slightly easier to read "
                    "using simpler words and shorter expressions."
                )

            elif direction == "complex":
                instruction = (
                    "Rewrite the sentence to make it slightly harder to read "
                    "using more complex words and expressions."
                )

            else:
                raise ValueError(
                    "For cnndm, direction must be "
                    "'simpler' or 'complex'."
                )

        elif task == "yelp":

            if direction == "positive":
                instruction = (
                    "Rewrite the sentence to make it "
                    "slightly more positive."
                )

            elif direction == "negative":
                instruction = (
                    "Rewrite the sentence to make it "
                    "slightly more negative."
                )

            else:
                raise ValueError(
                    "For yelp, direction must be "
                    "'positive' or 'negative'."
                )

        else:
            raise ValueError(
                "task must be 'cnndm' or 'yelp'."
            )

        prompt = (
            f'{instruction}\n\n'
            f'Sentence:\n'
            f'{sentence}\n\n'
            f'Return only the rewritten sentence.'
        )

        return prompt

    def build_sample_rewrite_prompt(
            self,
            sample,
            task,
            direction
    ):
        """
        Build full-sample rewriting prompt.
        """

        task = task.lower()
        direction = direction.lower()

        if task == "cnndm":

            if direction == "simpler":
                instruction = (
                    "Rewrite the text to make it slightly easier to read "
                    "using simpler words and shorter expressions."
                )

            elif direction == "complex":
                instruction = (
                    "Rewrite the text to make it slightly harder to read "
                    "using more complex words and expressions."
                )

            else:
                raise ValueError(
                    "For cnndm, direction must be "
                    "'simpler' or 'complex'."
                )

        elif task == "yelp":

            if direction == "positive":
                instruction = (
                    "Rewrite the text to make it "
                    "slightly more positive."
                )

            elif direction == "negative":
                instruction = (
                    "Rewrite the text to make it "
                    "slightly more negative."
                )

            else:
                raise ValueError(
                    "For yelp, direction must be "
                    "'positive' or 'negative'."
                )

        else:
            raise ValueError(
                "task must be 'cnndm' or 'yelp'."
            )

        prompt = (
            f"{instruction}\n\n"
            f"Text:\n"
            f"{sample}\n\n"
            f"Return only the rewritten text."
        )

        return prompt

    def build_sft_record(self, record):
        """Convert a selected sample into conversational SFT format."""

        prompt = self.build_sample_rewrite_prompt(
            sample=record["source_sample"],
            task=record["task"],
            direction=record["direction"]
        )

        return {
            "source_sample": record["source_sample"],
            "prompt": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": record["target_sample"]
                }
            ]
        }

    def export_sft_data(self, input_path, output_path):
        """Convert selected samples into SFT JSONL records."""

        import json

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with open(output_path, "w", encoding="utf-8") as f:
            for record in data:
                sft_record = self.build_sft_record(record)

                f.write(
                    json.dumps(sft_record, ensure_ascii=False) + "\n"
                )

        print(f"Exported {len(data)} SFT records to {output_path}")