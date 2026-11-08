import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader


class YelpDataset(Dataset):

    def __init__(
        self,
        data,
        tokenizer,
        text_col="text",
        label_col="label",
        max_length=256
    ):
        """
        data:
            pandas.DataFrame
            或 parquet

        tokenizer:
            HuggingFace tokenizer

        Yelp 原始标签:
            0 -> 1 star
            1 -> 2 stars
            2 -> 3 stars
            3 -> 4 stars
            4 -> 5 stars
        """

        self.data = data
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data[idx]

        text = str(row[0])
        label = int(row[1])

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": label
        }


class YelpDistilBERTClassifier(nn.Module):

    def __init__(
        self,
        model_name,
        num_labels=5,
        max_length=256
    ):
        super().__init__()

        self.max_length = max_length

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # DistilBERT + 标准分类头
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask, labels=None):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs

    def encode(self, texts):

        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return inputs

    @torch.no_grad()
    def predict(self, texts, device="cuda"):

        self.eval()
        self.to(device)

        inputs = self.encode(texts)

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

        pred_labels = torch.argmax(probs, dim=-1)

        return {
            "logits": logits.cpu(),
            "probs": probs.cpu(),
            "labels": pred_labels.cpu()
        }


class TrainProcess:
    def __init__(self):
        self.root_path = r"F:\python_code\StyleTrans2"
        self.model_name = r"F:\python_code\pre_model\distilbert-base-uncased"
        self.batch_size = 32
        self.epochs = 6
        self.lr = 2e-5

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        train_data = json.load(open(self.root_path + "/data/yelp_train_list.json", "r", encoding="utf-8"))
        train_dataset = YelpDataset(train_data, tokenizer)
        test_data = json.load(open(self.root_path + "/data/yelp_test_list.json", "r", encoding="utf-8"))
        test_dataset = YelpDataset(test_data, tokenizer)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=data_collator)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=data_collator)

        # batch = next(iter(train_loader))
        #
        # print(batch.keys())
        # print(batch["input_ids"].shape)
        # print(batch["attention_mask"].shape)
        # print(batch["labels"].shape)
        #
        # print(batch["labels"])

        model = YelpDistilBERTClassifier(model_name=self.model_name)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

        best_acc = 0.0
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            total_num = 0
            for batch in tqdm(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                # 清空上一轮梯度
                optimizer.zero_grad()
                # forward
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                # backward
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()
                total_loss += loss.item() * labels.size(0)
                total_num += labels.size(0)

            train_loss = total_loss / total_num

            # 9. Evaluation
            test_loss, test_acc = self.evaluate(
                model,
                test_loader,
                device
            )

            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Test Loss:  {test_loss:.4f}")
            print(f"Test Acc:   {test_acc:.4f}")

            # 10. Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                save_path = self.root_path+ "/model_pth/yelp_distilbert_best.pt"
                os.makedirs(os.path.dirname(save_path),exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved. Accuracy = {best_acc:.4f}")


    def evaluate(self, model, data_loader, device):
        model.eval()
        total_loss = 0.0
        total_num = 0
        correct = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total_loss += (loss.item() * labels.size(0))
                total_num += labels.size(0)

        avg_loss = total_loss / total_num
        accuracy = correct / total_num

        return avg_loss, accuracy


class YelpScorer:
    def __init__(self, model_path, checkpoint_path, device=None, max_length=256):
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.max_length = max_length

        self.model = None
        self.tokenizer = None

        self.load_model()

    def load_model(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True
        )

        self.model = YelpDistilBERTClassifier(
            model_name=self.model_path,
            num_labels=5
        )

        state_dict = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True
        )

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, text):

        single_input = isinstance(text, str)

        texts = [text] if single_input else text

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        logits = outputs.logits
        # [batch_size, 5]
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1) + 1
        levels = torch.arange(1, 6, device=self.device, dtype=probs.dtype)
        # [batch_size, 5] * [5]
        # -> [batch_size]
        scores = torch.sum(probs * levels, dim=-1)
        if single_input:
            return {
                "pred": preds[0].item(),
                "score": scores[0].item(),
                "probs": probs[0].cpu().tolist()
            }

        # list[str]
        return {
            "pred": preds.cpu().tolist(),
            "score": scores.cpu().tolist(),
            "probs": probs.cpu().tolist()
        }


if __name__ == "__main__":
    tp = TrainProcess()
    tp.train()
