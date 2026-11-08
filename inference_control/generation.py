import torch


class Generation:

    def __init__(self, model, processor, device="cuda"
    ):
        """
        Parameters
        ----------
        model :
            Hugging Face instruction-tuned causal language model.

        processor :
            Corresponding Hugging Face processor.
            The processor is responsible for applying the
            model-specific chat template.

        device : str
            Model device.
        """

        self.model = model
        self.processor = processor

        # Keep tokenizer for token length calculation
        self.tokenizer = processor.tokenizer

        self.device = device


    def generate(self, prompt, num_return_sequences=3, max_new_tokens=8, temperature=0.95,
        do_sample=True, num_beams=1):
        """
        Generate responses using the instruction-tuned model.

        Parameters
        ----------
        prompt : str
            User instruction.

        num_return_sequences : int
            Number of generated candidates.

        max_new_tokens : int
            Maximum number of newly generated tokens.

        temperature : float
            Sampling temperature.

        do_sample : bool
            Whether to use sampling.

        num_beams : int
            Number of beams when beam search is used.

        Returns
        -------
        list[str]
            Newly generated assistant responses only.
        """

        # --------------------------------------------------
        # 1. Convert normal prompt into chat messages
        # --------------------------------------------------

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        # --------------------------------------------------
        # 2. Apply Gemma-IT chat template
        # --------------------------------------------------

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = inputs.to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        # --------------------------------------------------
        # 3. Generation parameters
        # --------------------------------------------------

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": num_return_sequences,
            "do_sample": do_sample
        }

        if do_sample:

            generation_kwargs["temperature"] = temperature

        else:

            if num_beams < num_return_sequences:
                raise ValueError(
                    "When do_sample=False, num_beams must be "
                    "greater than or equal to num_return_sequences."
                )

            generation_kwargs["num_beams"] = num_beams

        # --------------------------------------------------
        # 4. Generate
        # --------------------------------------------------

        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,
                **generation_kwargs
            )

        # --------------------------------------------------
        # 5. Remove original chat input
        # --------------------------------------------------

        generated_tokens = outputs[:, input_len:]

        # --------------------------------------------------
        # 6. Decode assistant responses
        # --------------------------------------------------

        results = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return results

    def generate_batch(self, prompts, batch_size=8, num_return_sequences=3, max_new_tokens=8,
            temperature=0.95, do_sample=True, num_beams=1):
        """
        Batch version of generate().
        Parameters
        ----------
        prompts : list[str]
            Multiple user instructions.
        batch_size : int
            Number of prompts processed simultaneously.
        num_return_sequences : int
            Number of generated candidates for EACH prompt.
        max_new_tokens : int
            Maximum number of newly generated tokens.
        temperature : float
            Sampling temperature.
        do_sample : bool
            Whether to use sampling.
        num_beams : int
            Number of beams when beam search is used.
        Returns
        -------
        list[list[str]]
            [
                [output_0_0, output_0_1, output_0_2],
                [output_1_0, output_1_1, output_1_2],
                ...
            ]
        """

        if len(prompts) == 0:
            return []
        # Same constraint as original generate()
        if not do_sample:
            if num_beams < num_return_sequences:
                raise ValueError(
                    "When do_sample=False, num_beams must be "
                    "greater than or equal to num_return_sequences."
                )

        # 1. Decoder-only batch generation uses left padding
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        # Usually Gemma already has a pad token. Only fallback to EOS when none exists.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        all_results = []
        try:
            # 2. Process prompts by mini-batch
            for batch_start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[batch_start: batch_start + batch_size]
                # 3. Convert each prompt into one conversation
                batch_messages = []
                for prompt in batch_prompts:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]

                    batch_messages.append(messages)

                # batch_messages:
                # [
                #     [message_for_prompt_0],
                #     [message_for_prompt_1],
                #     ...
                # ]
                # 4. Apply Gemma-IT chat template in batch
                inputs = self.processor.apply_chat_template(
                    batch_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    processor_kwargs={
                        "padding": True
                    }
                )

                inputs = inputs.to(self.model.device)

                # All inputs are padded to the same length.
                input_len = inputs["input_ids"].shape[-1]

                # 5. Generation parameters
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "num_return_sequences": num_return_sequences,
                    "do_sample": do_sample
                }

                if do_sample:
                    generation_kwargs["temperature"] = temperature
                else:
                    generation_kwargs["num_beams"] = num_beams

                # 6. Generate
                with torch.inference_mode():
                    outputs = self.model.generate(**inputs, **generation_kwargs)

                # 7. Remove padded chat input
                generated_tokens = (outputs[:, input_len:])

                # 8. Decode
                decoded_outputs = (
                    self.processor.batch_decode(
                        generated_tokens,
                        skip_special_tokens=True
                    )
                )

                # Expected number:
                # current_batch_size * num_return_sequences
                expected_num = len(batch_prompts) * num_return_sequences

                if len(decoded_outputs) != expected_num:
                    raise RuntimeError(
                        f"Unexpected generation number: "
                        f"{len(decoded_outputs)} != {expected_num}"
                    )

                # 9. Group outputs back by prompt
                for i in range(len(batch_prompts)):
                    start = i * num_return_sequences
                    end = start + num_return_sequences
                    all_results.append(decoded_outputs[start:end])

        finally:
            # Restore tokenizer setting
            self.tokenizer.padding_side = old_padding_side

        return all_results