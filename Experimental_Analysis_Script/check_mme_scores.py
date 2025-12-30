import os

BASE_DIR = "/home/lsy/workspace/opensourceToolKit/VLMEvalKit/workdir/pasta_mm/llava_v1.5_7b/single_head_reweight"

missing = []

for layer in range(27):      # 0-26
    for head in range(32):   # 0-31
        folder = f"llava_pastamm_{layer}_{head}"
        csv_name = f"{folder}_MME_score.csv"
        csv_path = os.path.join(BASE_DIR, folder, csv_name)

        if not os.path.exists(csv_path):
            missing.append((layer, head))

# ===== 输出结果 =====
if not missing:
    print("[OK] All CSV files exist.")
else:
    print(f"[WARN] Missing {len(missing)} CSV files:")
    for layer, head in missing:
        print(f"  - layer={layer}, head={head}")