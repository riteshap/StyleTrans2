
import os
import json

task = "yelp"
cur_path = os.getcwd()
print(cur_path)
ret_train_data = []
for file_name in os.listdir(cur_path):
    if task not in file_name:
        continue
    if "sft-traindata" not in file_name:
        continue
    if not file_name.endswith(".json"):
        continue
    if len(file_name.split("_")) != 4:
        continue

    print(file_name)
    file  = json.load(open(os.path.join(cur_path, file_name), "r", encoding="utf-8"))
    ret_train_data.extend(file)
    ret_train_data = sorted(ret_train_data, key=lambda x: (x['source_style_level'], x['sample_id']))

save_path = os.path.join(cur_path, f"{task}_ori_traindata.json")
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(ret_train_data, f, ensure_ascii=False, indent=2)