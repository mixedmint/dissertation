import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data processing/7_regression.csv", encoding="utf-8-sig")

# ── 选取诊断变量 ──
# 参照组：ethnicity=White，business_type=restaurant_cafe_canteen（已删去）
# age：保留 0-4 和 65+
age_cols          = ["0-4", "65plus"]
ethnicity_cols    = ["Asian", "Black", "Other"]
business_type_cols = [
    "retailers___other",
    "takeaway_sandwich_shop",
    "pub_bar_nightclub",
    "supermarkets_hypermarkets",
    "hotel_bed_and_breakfast_guest_house",
    "other_catering_premises",
]
control_cols   = age_cols + ethnicity_cols + business_type_cols + ["Population_density"]
all_vars       = ["IMD"] + control_cols

# 检查列名是否存在
missing = [c for c in all_vars if c not in df.columns]
if missing:
    print(f"警告：以下列不存在，请核查列名：{missing}")
    all_vars = [c for c in all_vars if c in df.columns]

X = df[all_vars].dropna()
print(f"样本量：{len(X)}（删除缺失值后）\n")

# ── 1. 两两相关系数矩阵 ──
corr = X.corr().round(3)
print("=" * 60)
print("两两相关系数矩阵")
print("=" * 60)
print(corr.to_string())
corr.to_csv("NBR/with_age/8_correlation_matrix_with_age.csv", encoding="utf-8-sig")

# 热力图
LABEL_MAP = {
    "IMD":                              "IMD",
    "0-4":                              "Age 0-4",
    "65plus":                           "Age 65+",
    "Asian":                            "Asian",
    "Black":                            "Black",
    "Other":                            "Other",
    "Population_density":               "Population Density",
    "retailers___other":                "Other Retailers",
    "takeaway_sandwich_shop":           "Takeaway & Sandwich Shops",
    "pub_bar_nightclub":                "Pubs, Bars & Night Clubs",
    "supermarkets_hypermarkets":        "Supermarkets & Hypermarkets",
    "hotel_bed_and_breakfast_guest_house": "Hotels & Guesthouses",
    "other_catering_premises":          "Other Caters",
}
labels = [LABEL_MAP.get(c, c) for c in corr.columns]

n = len(corr)
fig, ax = plt.subplots(figsize=(n * 0.7 + 1, n * 0.7 + 1))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                fontsize=6, color="black" if abs(corr.values[i, j]) < 0.7 else "white")
ax.set_title("Correlation Matrix", fontsize=12, pad=12)
plt.tight_layout()
plt.savefig("NBR/with_age/8_correlation_heatmap_with_age.png", dpi=150)
plt.close()
print("热力图已保存：NBR/with_age/8_correlation_heatmap_with_age.png")

# ── 2. VIF ──
print("\n" + "=" * 60)
print("VIF（方差膨胀因子）")
print("=" * 60)
print("参照组：ethnicity=White，business_type=restaurant_cafe_canteen（已删去）\n")

vif_data = pd.DataFrame({
    "Variable": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
vif_data["VIF"] = vif_data["VIF"].round(2)
vif_data = vif_data.sort_values("VIF", ascending=False)
print(vif_data.to_string(index=False))
vif_data.to_csv("NBR/with_age/8_vif_with_age.csv", index=False, encoding="utf-8-sig")

print("\n结果已保存：8_correlation_matrix_with_age.csv、8_vif_with_age.csv")
