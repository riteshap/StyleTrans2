from sft_data import SFTDataModule
from sft_model import SFTModel


def main():
    task = "yelp"
    cur_path = r"/root/private_data/StyleTrans2/sft"

    llm_name = "gemma-4-12B-it"
    llm_dict = {"gemma-4-12B-it": "ge12B",
                "gemma-4-E2B-it": "geE2B",
                "Phi-4-mini-instruct": "phi4mini",
                "Phi-4": "phi4"}

    # Initialize the dataset module.
    data_module = SFTDataModule(
        task=task,
        data_path=cur_path + f"/{task}_sft_formatted.jsonl",
        output_dir="/data",
        val_ratio=0.1,
        seed=42
    )

    # Initialize the training model.
    sft_model = SFTModel(
        model_path=f"/root/private_data/{llm_name}",
        data_module=data_module,
        output_dir=f"outputs/{task}_{llm_dict[llm_name]}_sft",

        model_type="vision_language", # "causal_lm",
        use_4bit=False,

        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,

        learning_rate=1e-4,
        num_train_epochs=3,

        batch_size=4,
        gradient_accumulation_steps=16,

        max_length=2048,
        seed=42
    )

    # Initialize the trainer.
    sft_model.build_trainer()

    # Execute training.
    sft_model.train()

    # Save the trained adapter.
    sft_model.save_model()


if __name__ == "__main__":
    main()