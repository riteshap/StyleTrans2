import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import json
import textstat
import torch
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from yelp_classification.yelp_classification import YelpScorer
import nltk

nltk.data.path.insert(0, "/root/private_data/nltk_data")


class TextEvaluator:
    def __init__(self, task, root_path, llm_name, llm_dict, sbert_model_path, threshold=0.6, device="cuda",
                 enable_yelp=False, yelp_model_path=None, yelp_checkpoint_path=None, yelp_max_length=256):
        """
        Parameters
        sbert_model_path : str  Local Sentence-BERT model path.
        device : str  "cuda" or "cpu"
        """
        assert task in ["yelp", "cnndm"]
        self.task = task
        self.root_path = root_path
        self.cur_path = root_path + "/inference_control"
        self.llm_name = llm_name
        self.llm_dict = llm_dict
        self.s_llm_name = self.llm_dict[llm_name]
        self.threshold = threshold
        self.device = device
        self.semantic_model = SentenceTransformer(sbert_model_path, device=device)
        self.semantic_model.eval()

        self.enable_yelp = enable_yelp
        self.yelp_scorer = None

        if enable_yelp:
            if yelp_model_path is None:
                raise ValueError("yelp_model_path must be provided when enable_yelp=True.")
            if yelp_checkpoint_path is None:
                raise ValueError("yelp_checkpoint_path must be provided when enable_yelp=True.")
            self.yelp_scorer = YelpScorer(model_path=yelp_model_path, checkpoint_path=yelp_checkpoint_path,
                device=device, max_length=yelp_max_length)

    def calculate_similarity(self, source_text, generated_text):
        """
        Calculate semantic similarity between
        source and generated sentence using SBERT.
        Returns cosine similarity.
        """

        if (
            not source_text
            or not generated_text
            or not source_text.strip()
            or not generated_text.strip()
        ):
            return None
        texts = [source_text.strip(), generated_text.strip()]
        with torch.inference_mode():
            embeddings = self.semantic_model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
        similarity = cos_sim(embeddings[0], embeddings[1]).item()
        return float(similarity)

    def calculate_fre(self, text: str):
        """
        Calculate Flesch Reading Ease score.
        Higher score = easier to read.
        """
        if not text or not text.strip():
            return None
        score = textstat.flesch_reading_ease(text.strip())
        return float(score)

    def calculate_yelp_sentiment(self, text: str):
        """
        Calculate continuous Yelp sentiment score.
        Score range: approximately 1.0 ~ 5.0
        Calculation: score = 1 * P(class=1) + 2 * P(class=2) + 3 * P(class=3) + 4 * P(class=4) + 5 * P(class=5)
        Parameters
        text : str Single sentence or complete Yelp review.
        Returns
        float Continuous sentiment score.
        """
        if not self.enable_yelp:
            raise RuntimeError("Yelp scorer is not enabled. Initialize TextEvaluator with enable_yelp=True.")
        if not isinstance(text, str):
            raise TypeError("calculate_yelp_sentiment() expects a single str.")
        if not text.strip():
            return None
        result = self.yelp_scorer.score(text.strip())
        return float(result["score"])

    def calculate_style_score(self, text):
        if self.task == "yelp":
            return self.calculate_yelp_sentiment(text)
        elif self.task == "cnndm":
            return self.calculate_fre(text)
        return None

    def main(self):
        edits_path = f"{self.cur_path}/{task}_{self.s_llm_name}_edits.json"
        with open(edits_path, "r", encoding="utf-8") as f:
            file = json.load(f)

        save_path = f"{self.cur_path}/{task}_{self.s_llm_name}_evaluated.json"
        if os.path.exists(save_path):
            return None

        total_out = []
        for sample_ele in file:
            if len(sample_ele) == 0:
                continue
            sample_out = []
            style_level = sample_ele[0]["style_level"]
            sample_id = sample_ele[0]["sample_id"]
            print(f"style_level:{style_level}, sample_id: {sample_id}")
            for sent_ele in sample_ele:
                style_level = sent_ele["style_level"]
                sample_id = sent_ele["sample_id"]
                sent_id = sent_ele["sent_id"]
                source_sentence = sent_ele["source_sentence"]

                source_style = self.calculate_style_score(source_sentence)
                direction = sent_ele["direction"]
                mode = sent_ele["mode"]
                start_char = sent_ele.get("start_char", None)
                end_char = sent_ele.get("end_char", None)
                original_span = sent_ele.get("original_span", None)
                if mode in ["fill", "replace"]:
                    generated_sentences = sent_ele["restored_sentence"]
                    generated_span = sent_ele["generated_span"]
                else:
                    # rewrite
                    generated_span = None
                    generated_sentences = sent_ele["raw_output"]

                unique_sentences = []
                unique_spans = []
                seen = set()
                for temp_ind, sent in enumerate(generated_sentences):
                    if not isinstance(sent, str):
                        continue
                    sent = sent.strip()
                    if len(sent) == 0:
                        continue
                    if sent not in seen:
                        seen.add(sent)
                        unique_sentences.append(sent)
                        if generated_span is not None:
                            unique_spans.append(generated_span[temp_ind])
                            assert generated_span[temp_ind] in sent

                # Evaluate each candidate
                valid_candidates = []
                for temp_ind, generated_sentence in enumerate(unique_sentences):
                    semantic = self.calculate_similarity(source_sentence, generated_sentence)
                    # semantic filter
                    if semantic < self.threshold:
                        continue
                    generated_style = self.calculate_style_score(generated_sentence)
                    delta_style = generated_style - source_style
                    unique_span = unique_spans[temp_ind] if len(unique_spans) > 0 else None

                    valid_candidates.append(
                        {
                            "style_level": style_level,
                            "sample_id": sample_id,
                            "sent_id": sent_id,
                            "mode": mode,
                            "direction": direction,
                            "start_char": start_char,
                            "end_char": end_char,
                            "original_span": original_span,
                            "generated_span": unique_span,
                            "source_sentence": source_sentence,
                            "generated_sentence": generated_sentence,
                            "source_style_score": source_style,
                            "generated_style_score": generated_style,
                            "delta_style": delta_style,
                            "semantic_similarity": semantic
                        }
                    )

                sample_out.extend(valid_candidates)
            total_out.append(sample_out)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(total_out, f, ensure_ascii=False, indent=4)

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

    te.main()
