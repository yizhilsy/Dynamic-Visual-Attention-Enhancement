import os
import pandas as pd

# ===== 路径配置 =====
BASE_DIR = "/home/lsy/workspace/opensourceToolKit/VLMEvalKit/workdir/pasta_mm/llava_v1.5_7b/single_head_reweight"
BASELINE_CSV = "/home/lsy/workspace/opensourceToolKit/VLMEvalKit/workdir/pasta_mm/llava_v1.5_7b/bf16baseline/llava_v1.5_7b_MME_score.csv"

OUTPUT_RAW_CSV = "./Experimental_Data/llava_pastamm_mme_all_0-21.csv"
OUTPUT_GAIN_CSV = "./Experimental_Data/llava_pastamm_mme_gain_vs_baseline_0-21.csv"

# ===== 读取 baseline =====
baseline_df = pd.read_csv(BASELINE_CSV)
baseline = baseline_df.iloc[0].copy()

# ✅ 新增 baseline 的 mme_total_score
baseline["mme_total_score"] = baseline["perception"] + baseline["reasoning"]

all_rows = []

# ===== 收集 single-head 结果 =====
for layer in range(22):          # 0-21
    for head in range(32):      # 0-31
        folder = f"llava_pastamm_{layer}_{head}"
        csv_name = f"{folder}_MME_score.csv"
        csv_path = os.path.join(BASE_DIR, folder, csv_name)

        if not os.path.exists(csv_path):
            print(f"[WARN] Missing: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # ===== 新增 mme_total_score =====
        if "perception" not in df or "reasoning" not in df:
            raise KeyError("Missing perception or reasoning column")

        df["mme_total_score"] = df["perception"] + df["reasoning"]

        # 添加 layer / head
        df.insert(0, "head", head)
        df.insert(0, "layer", layer)

        all_rows.append(df)

# ===== 合并原始结果 =====
final_df = pd.concat(all_rows, ignore_index=True)
final_df.to_csv(OUTPUT_RAW_CSV, index=False)
print(f"[OK] Saved raw results to {OUTPUT_RAW_CSV}")

# ===== 计算相对 baseline 的涨幅 =====
gain_df = final_df[["layer", "head"]].copy()

metric_cols = [
    col for col in final_df.columns
    if col not in ["layer", "head"]
]

for col in metric_cols:
    if col not in baseline:
        print(f"[WARN] Metric {col} not in baseline, skipped.")
        continue

    gain_df[col + "_gain"] = (final_df[col] - baseline[col]) / baseline[col]

# ===== 保存涨幅 CSV =====
gain_df.to_csv(OUTPUT_GAIN_CSV, index=False)
print(f"[OK] Saved gain results to {OUTPUT_GAIN_CSV}")