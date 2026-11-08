
import json
import torch

from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)

from trl import SFTTrainer, SFTConfig


class SFTModel:

    def __init__(
        self,
        model_path,
        data_module,
        output_dir,
        model_type="vision_language",
        use_4bit=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=None,
        learning_rate=1e-4,
        num_train_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=16,
        max_length=2048,
        seed=42
    ):
        """
        Initialize the LoRA-SFT training model.

        Args:
            model_path: Local path or Hugging Face model ID.
            data_module: An initialized SFTDataModule instance.
            output_dir: Directory for checkpoints and adapters.
            model_type: "causal_lm" or "vision_language".
            use_4bit: Enable 4-bit QLoRA training.
            lora_r: LoRA rank.
            lora_alpha: LoRA scaling parameter.
            lora_dropout: LoRA dropout rate.
            target_modules: Transformer modules adapted by LoRA.
            learning_rate: Training learning rate.
            num_train_epochs: Number of training epochs.
            batch_size: Per-device training batch size.
            gradient_accumulation_steps: Gradient accumulation.
            max_length: Maximum training sequence length.
            seed: Random seed.
        """

        self.model_path = str(model_path)
        self.data_module = data_module
        self.output_dir = Path(output_dir)

        self.model_type = model_type
        self.use_4bit = use_4bit

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        self.target_modules = target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ]

        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        self.seed = seed

        self.model = None
        self.tokenizer = None
        self.processor = None
        self.trainer = None

        if model_type not in ("causal_lm", "vision_language"):
            raise ValueError(
                "model_type must be 'causal_lm' or 'vision_language'."
            )

        if use_4bit and not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit QLoRA training requires a CUDA GPU."
            )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def load_model(self):
        """
        Load the base model and its tokenizer or processor.
        """

        print(f"Loading model: {self.model_path}")

        # Select model precision.
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        else:
            dtype = torch.float32

        model_kwargs = {
            "dtype": dtype
        }

        # Optional 4-bit quantization.
        if self.use_4bit:

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype
            )

            model_kwargs["quantization_config"] = (
                quantization_config
            )

            # Single-GPU QLoRA configuration.
            model_kwargs["device_map"] = {
                "": torch.cuda.current_device()
            }

        # Load the appropriate model architecture.
        if self.model_type == "causal_lm":

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path
            )

            self.processor = self.tokenizer

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )

        else:

            self.processor = AutoProcessor.from_pretrained(
                self.model_path
            )

            self.tokenizer = self.processor.tokenizer

            self.model = (
                AutoModelForImageTextToText.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
            )

        if self.tokenizer.chat_template is None:
            raise ValueError(
                "The tokenizer does not provide a chat template."
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        # Disable KV caching during gradient checkpointing.
        self.model.config.use_cache = False

        if hasattr(self.model.config, "text_config"):
            self.model.config.text_config.use_cache = False

        # Prepare quantized weights for training.
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(
                self.model
            )

        print("Model loaded successfully.")

        return self.model

    def build_lora_config(self):
        """Create a LoRA configuration for the language model."""

        if self.model is None:
            raise RuntimeError("Model must be loaded first.")

        if self.model.config.model_type.startswith("gemma4"):

            # Target only the language model attention projections.
            target_modules = (
                r".*\.language_model\.layers\.\d+"
                r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
            )

        else:

            # Default targets for standard causal language models.
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj"
            ]

        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

    def build_trainer(self):
        """
        Prepare datasets and initialize the SFTTrainer.
        """

        if self.model is None:
            self.load_model()

        # Load training and validation datasets.
        train_dataset, val_dataset = (
            self.data_module.prepare()
        )

        lora_config = self.build_lora_config()

        use_bf16 = (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )

        use_fp16 = (
            torch.cuda.is_available()
            and not use_bf16
        )

        training_args = SFTConfig(
            output_dir=str(self.output_dir),

            # Training parameters
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=(
                self.gradient_accumulation_steps
            ),

            learning_rate=self.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=0.03,
            weight_decay=0.01,
            max_grad_norm=1.0,

            # Precision
            bf16=use_bf16,
            fp16=use_fp16,

            # Memory optimization
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            },

            # SFT configuration
            max_length=self.max_length,
            completion_only_loss=True,
            assistant_only_loss=False,
            packing=False,

            # Evaluation and checkpointing
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,

            # Logging
            logging_steps=20,
            report_to="none",

            seed=self.seed,
            data_seed=self.seed
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.processor,
            peft_config=lora_config
        )

        print("SFTTrainer initialized.")

        self.trainer.model.print_trainable_parameters()

        return self.trainer

    def check_training_batch(self):
        """
        Check that supervised target tokens are available.
        """

        if self.trainer is None:
            raise RuntimeError(
                "The SFTTrainer has not been initialized."
            )

        batch = next(
            iter(self.trainer.get_train_dataloader())
        )

        labels = batch["labels"]

        target_tokens = int(
            (labels != -100).sum().item()
        )

        masked_tokens = int(
            (labels == -100).sum().item()
        )

        print(f"Supervised tokens: {target_tokens}")
        print(f"Masked tokens: {masked_tokens}")

        if target_tokens == 0:
            raise RuntimeError(
                "No supervised tokens found in the training batch."
            )

    def train(self, resume_from_checkpoint=None):
        """
        Execute LoRA-SFT training.

        Args:
            resume_from_checkpoint:
                Optional checkpoint path for resuming training.
        """

        if self.trainer is None:
            self.build_trainer()

        self.check_training_batch()

        print("Starting LoRA-SFT training...")

        train_result = self.trainer.train(
            resume_from_checkpoint=resume_from_checkpoint
        )

        print("Training completed.")

        return train_result

    def save_model(self):
        """
        Save the trained LoRA adapter and training metadata.
        """

        if self.trainer is None:
            raise RuntimeError(
                "No trainer is available for saving."
            )

        adapter_dir = self.output_dir / "final_adapter"

        adapter_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save the trained LoRA adapter.
        self.trainer.save_model(str(adapter_dir))

        # Save tokenizer or processor files.
        self.processor.save_pretrained(str(adapter_dir))

        # Save model metadata for future inference.
        metadata = {
            "base_model": self.model_path,
            "model_type": self.model_type,
            "use_4bit": self.use_4bit,
            "max_length": self.max_length,
            "seed": self.seed
        }

        with open(
            adapter_dir / "model_info.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                metadata,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(f"LoRA adapter saved to: {adapter_dir}")

        return adapter_dir
