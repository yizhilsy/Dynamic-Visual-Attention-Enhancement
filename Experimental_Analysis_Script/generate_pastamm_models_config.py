import json

NUM_LAYERS = 32
SELECTED_HEADS = [0, 8, 16, 24, 31]  # 只选这几个 head

MODEL_CLASS = "LLaVA_PastaMM"
MODEL_PATH = "/s/llava-series/llava-v1.5-7b"

# config = {
#     "model": {},
#     "data": {
#         "POPE": {
#             "class": "ImageYORNDataset",
#             "dataset": "POPE"
#         }
#     }
# }

config = {
    "model": {},
    "data": {
        "ScienceQA_TEST": {
            "class": "ImageMCQDataset",
            "dataset": "ScienceQA_TEST"
        }
    }
}

for layer in range(NUM_LAYERS):
    for head in SELECTED_HEADS:
        model_name = f"llava_pastamm_{layer}_{head}"
        config["model"][model_name] = {
            "class": MODEL_CLASS,
            "model_path": MODEL_PATH,
            "head_config": {
                str(layer): [head]  # key 用字符串
            }
        }

# 输出为 JSON 文件
# output_path = "llava_pastamm_selected_heads_eval_pope.json"
output_path = "./Experimental_Data/llava_pastamm_selected_heads_eval_sqa.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4, ensure_ascii=False)

print(f"Generated config file: {output_path}")
print(f"Total models: {NUM_LAYERS * len(SELECTED_HEADS)}")