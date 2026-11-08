
import json
import random
from pathlib import Path
from collections import defaultdict

from datasets import Dataset


class SFTDataModule:

    def __init__(self, task, data_path, output_dir, val_ratio=0.1, seed=42):
        """
        Initialize the SFT data module.

        Args:
            data_path: Path to the formatted SFT dataset.
            output_dir: Directory for saving train and validation data.
            val_ratio: Proportion of validation samples.
            seed: Random seed for reproducible splitting.
        """
        assert task in ["yelp", "cnndm"]
        self.task = task
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)

        self.val_ratio = val_ratio
        self.seed = seed

        self.data = None
        self.train_data = None
        self.val_data = None

        if not 0 < val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1.")

        if not self.data_path.is_file():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

    def load_data(self):
        """Load formatted SFT records from JSON or JSONL."""

        if self.data_path.suffix.lower() == ".json":
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        elif self.data_path.suffix.lower() == ".jsonl":
            data = []
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            raise ValueError("Unsupported file format. Use JSON or JSONL.")

        if not isinstance(data, list) or not data:
            raise ValueError("Dataset must be a non-empty list.")
        self._validate_data(data)
        self.data = data
        print(f"Loaded {len(data)} SFT records.")

        return data

    def _validate_data(self, data):
        """Validate the conversational prompt-completion format."""

        for index, record in enumerate(data):
            if not isinstance(record, dict):
                raise ValueError(f"Record {index}: Invalid record type.")

            prompt = record.get("prompt")
            completion = record.get("completion")

            if not isinstance(prompt, list) or not prompt:
                raise ValueError(f"Record {index}: Invalid prompt.")

            if not isinstance(completion, list) or len(completion) != 1:
                raise ValueError(f"Record {index}: Invalid completion.")

            for message in prompt + completion:
                if not isinstance(message, dict):
                    raise ValueError(f"Record {index}: Invalid message structure.")

                if not isinstance(message.get("content"), str):
                    raise ValueError(f"Record {index}: Message content must be a string.")

                if not message["content"].strip():
                    raise ValueError(f"Record {index}: Empty message content.")

            if prompt[-1].get("role") != "user":
                raise ValueError(f"Record {index}: Prompt must end with a user message.")

            if completion[0].get("role") != "assistant":
                raise ValueError(f"Record {index}: Completion role must be assistant.")

            # Verify that the source text can be identified.
            self._get_source_text(record)

    def _get_source_text(self, record):
        """
        Extract the source text for group-based splitting.

        Prefer explicit source_sample metadata.
        Otherwise, recover it from the existing rewrite prompt.
        """

        source = record.get("source_sample")

        if isinstance(source, str) and source.strip():
            return source
        else:
            raise Exception("loss key: source_sample")

    def split_dataset(self):
        """
        Split records into training and validation sets.

        Records sharing the same source text are kept
        in the same split to prevent data leakage.
        """

        if self.data is None:
            self.load_data()

        groups = defaultdict(list)

        for record in self.data:
            source = self._get_source_text(record)
            groups[source].append(record)

        group_keys = list(groups.keys())

        if len(group_keys) < 2:
            raise ValueError("At least two distinct source samples are required.")

        rng = random.Random(self.seed)
        rng.shuffle(group_keys)

        val_group_count = round(len(group_keys) * self.val_ratio)

        val_group_count = max(1, min(val_group_count, len(group_keys) - 1))

        val_keys = set(group_keys[:val_group_count])

        self.train_data = []
        self.val_data = []

        for source, records in groups.items():

            if source in val_keys:
                self.val_data.extend(records)
            else:
                self.train_data.extend(records)

        print("\n========== Dataset Split ==========")
        print(f"Total records : {len(self.data)}")
        print(f"Train records : {len(self.train_data)}")
        print(f"Val records   : {len(self.val_data)}")
        print(f"Random seed   : {self.seed}")

        return self.train_data, self.val_data

    def save_splits(self):
        """Save training and validation records as JSONL."""

        if self.train_data is None or self.val_data is None:
            raise RuntimeError("Dataset has not been split.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        output_files = {
            f"{self.task}_sft_train.jsonl": self.train_data,
            f"{self.task}_sft_val.jsonl": self.val_data
        }

        for filename, records in output_files.items():

            output_path = self.output_dir / filename

            with open(output_path, "w", encoding="utf-8") as f:

                for record in records:

                    # Only retain fields required by SFTTrainer.
                    sft_record = {
                        "prompt": record["prompt"],
                        "completion": record["completion"]
                    }
                    f.write(json.dumps(sft_record, ensure_ascii=False) + "\n")

            print(f"Saved: {output_path}")

    def get_datasets(self):
        """
        Convert the split data into Hugging Face Dataset objects.

        Returns:
            train_dataset, val_dataset
        """

        if self.train_data is None or self.val_data is None:
            raise RuntimeError("Dataset has not been split.")

        train_records = [
            {
                "prompt": record["prompt"],
                "completion": record["completion"]
            }
            for record in self.train_data
        ]

        val_records = [
            {
                "prompt": record["prompt"],
                "completion": record["completion"]
            }
            for record in self.val_data
        ]

        train_dataset = Dataset.from_list(train_records)
        val_dataset = Dataset.from_list(val_records)

        return train_dataset, val_dataset

    def prepare(self):
        """Execute the complete SFT data preparation pipeline."""

        self.load_data()
        self.split_dataset()
        self.save_splits()

        return self.get_datasets()

if __name__ == "__main__":
    task="yelp"
    cur_path = r"F:\python_code\StyleTrans2\sft"
    data_module = SFTDataModule(
        task=task,
        data_path=cur_path + f"/{task}_sft_formatted.jsonl",
        output_dir=cur_path + f"/data",
        val_ratio=0.1,
        seed=42
    )

    train_dataset, val_dataset = data_module.prepare()