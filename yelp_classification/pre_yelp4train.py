import os
import json

root_path = r"F:\python_code\StyleTrans2"
for v_t in ["train", "test"]:
    test_data = json.load(open(os.path.join(root_path, f"data/yelp_{v_t}.json")))
    test_list = []
    for key_ind in test_data.keys():
        label = int(key_ind)
        for text in test_data[key_ind]:
            test_list.append((text, label))
    json.dump(test_list, open(os.path.join(root_path, f"data/yelp_{v_t}_list.json"), "w"), ensure_ascii=False, indent=2)

