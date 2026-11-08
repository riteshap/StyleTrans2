import os
import json
import time
import torch
from Evaluator import TextEvaluator

class SampleRewrite:
    def __init__(self, task, root_path:str, llm_root_path:str, llm_name:str, llm_dict:dict, mode:str="base"):
        assert task in ["yelp", "cnndm"]
        self.task = task
        self.root_path = root_path
        self.cur_path = root_path + "/inference_control"
        self.llm_root_path = llm_root_path
        self.llm_path = f"{llm_root_path}/{llm_name}"
        self.llm_name = llm_name
        self.llm_dict = llm_dict
        self.s_llm_name = self.llm_dict[llm_name]
        self.mode = mode
        assert self.mode in ["base", "sft"]

    def evaluator(self, te:TextEvaluator):
        save_path = self.cur_path + f"/{self.task}_{self.s_llm_name}_{self.mode}_sample_rewrite_v2.json"
        if os.path.exists(save_path):
            print(f"Sample rewrite file already exists: {save_path}")
            return None

        file_path = self.cur_path + f"/{self.task}_{self.s_llm_name}_{self.mode}_sample_rewrite.json"
        data_file = json.load(open(file_path, "r", encoding="utf-8"))

        for dir_list in data_file:
            for ele in dir_list:
                if ele["sample_id"] % 10 == 0:
                    print(f"style_level:{ele["style_level"]}; sample_id:{ele['sample_id']}")
                source_text = ele["source_sample"]
                raw_output_list = ele["raw_output"]
                source_text_style_score = te.calculate_style_score(source_text)
                sim_score_list = []
                style_score_list = []
                for gen_text in raw_output_list:
                    text_style_score = te.calculate_style_score(gen_text)
                    sim_score = te.calculate_similarity(source_text, gen_text)
                    style_score_list.append(text_style_score)
                    sim_score_list.append(sim_score)

                ele["source_style_score"] = source_text_style_score
                ele["sim_score"] = sim_score_list
                ele["style_score"] = style_score_list

        json.dump(data_file, open(save_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return None


if __name__ == "__main__":
    task = "yelp"
    mode = "sft"
    root_path = r"F:\python_code\StyleTrans2"
    llm_root_path = r"F:\python_code\pre_model"
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

    if mode == "base":
        sr_class = SampleRewrite(task, root_path, llm_root_path, llm_name, llm_dict)
        sr_class.evaluator(te)
    else:
        sr_class = SampleRewrite(task, root_path, llm_root_path, llm_name, llm_dict, mode=mode)
        sr_class.evaluator(te)



