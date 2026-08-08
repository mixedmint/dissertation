import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ── 第一步（人工完成）：按 full correlation matrix，以 |r| > 0.5 为阈值删除变量 ──
# 删除：4_15, 25-44, 45-64, 65plus, White, Mixed_or_Multiple, Black, restaurant_cafe_canteen
# （已核实：剩余13个变量两两相关系数全部 < 0.5，详见 NBR/8_full_correlation_matrix.csv）
#
# ── 第二步（本脚本）：对剩余变量 + IMD 做逐步 VIF 剔除，阈值 10，IMD 强制保留 ──
VIF_THRESHOLD = 10
PROTECTED = "IMD"

df = pd.read_csv("data processing/7_regression.csv", encoding="utf-8-sig")

all_vars = [
    PROTECTED,
    "0-4", "15-19", "20-24",
    "Asian", "Other",
    "retailers___other", "takeaway_sandwich_shop", "pub_bar_nightclub",
    "other_catering_premises", "supermarkets_hypermarkets",
    "hotel_bed_and_breakfast_guest_house",
    "Population_density",
]

missing = [c for c in all_vars if c not in df.columns]
if missing:
    print(f"警告：以下列不存在，请核查列名：{missing}")
    all_vars = [c for c in all_vars if c in df.columns]

X = df[all_vars].dropna()
print(f"相关系数筛选后起始变量数：{len(all_vars)}（含 IMD）")
print(f"样本量：{len(X)}\n")

current_vars = list(all_vars)
history_rows = []
step = 0

while True:
    Xi = X[current_vars].values
    vif_vals = {
        col: variance_inflation_factor(Xi, i)
        for i, col in enumerate(current_vars)
    }
    vif_series = pd.Series(vif_vals).sort_values(ascending=False)

    step += 1
    print(f"── 第 {step} 轮（剩余 {len(current_vars)} 个变量）──")
    print(vif_series.round(2).to_string())
    print()

    for var, vif in vif_series.items():
        history_rows.append({"Step": step, "Variable": var, "VIF": round(vif, 4)})

    candidates = vif_series.drop(index=PROTECTED, errors="ignore")
    if candidates.empty or candidates.max() <= VIF_THRESHOLD:
        break

    drop_var = candidates.idxmax()
    print(f">>> 删除变量：{drop_var}（VIF={candidates.max():.2f} > {VIF_THRESHOLD}）\n")
    current_vars.remove(drop_var)

print("=" * 60)
print(f"最终保留变量（全部 VIF < {VIF_THRESHOLD}，IMD 强制保留）")
print("=" * 60)
final_vif = pd.Series(vif_vals).sort_values(ascending=False)
print(final_vif.round(2).to_string())

history_df = pd.DataFrame(history_rows)
history_df.to_csv("NBR/vif_stepwise_after_corr_filter_history.csv", index=False, encoding="utf-8-sig")

final_df = final_vif.reset_index()
final_df.columns = ["Variable", "VIF"]
final_df["VIF"] = final_df["VIF"].round(4)
final_df.to_csv("NBR/vif_stepwise_after_corr_filter_final.csv", index=False, encoding="utf-8-sig")

removed = [v for v in all_vars if v not in current_vars]
print(f"\n共删除 {len(removed)} 个变量：{removed}")
print("\n结果已保存：")
print("  NBR/vif_stepwise_after_corr_filter_history.csv")
print("  NBR/vif_stepwise_after_corr_filter_final.csv")
