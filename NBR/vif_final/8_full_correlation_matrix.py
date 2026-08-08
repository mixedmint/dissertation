import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data processing/7_regression.csv", encoding="utf-8-sig")

# ── 全部控制变量（不做参照组删减，纯描述性相关矩阵）──
age_cols        = ["0-4", "4_15", "15-19", "20-24", "25-44", "45-64", "65plus"]
ethnicity_cols   = ["Asian", "Black", "Mixed_or_Multiple", "White", "Other"]
business_type_cols = [
    "restaurant_cafe_canteen", "retailers___other", "takeaway_sandwich_shop",
    "pub_bar_nightclub", "other_catering_premises",
    "supermarkets_hypermarkets", "hotel_bed_and_breakfast_guest_house",
]
all_vars = ["IMD"] + age_cols + ethnicity_cols + business_type_cols + ["Population_density"]

missing = [c for c in all_vars if c not in df.columns]
if missing:
    print(f"警告：以下列不存在，请核查列名：{missing}")
    all_vars = [c for c in all_vars if c in df.columns]

X = df[all_vars].dropna()
print(f"样本量：{len(X)}（删除缺失值后）")
print(f"变量数：{len(all_vars)}\n")

corr = X.corr().round(3)
corr.to_csv("NBR/vif_final/8_full_correlation_matrix.csv", encoding="utf-8-sig")

LABEL_MAP = {
    "IMD":                              "IMD",
    "0-4":                              "Age 0-4",
    "4_15":                             "Age 5-14",
    "15-19":                            "Age 15-19",
    "20-24":                            "Age 20-24",
    "25-44":                            "Age 25-44",
    "45-64":                            "Age 45-64",
    "65plus":                           "Age 65+",
    "Asian":                            "Asian",
    "Black":                            "Black",
    "Mixed_or_Multiple":                "Mixed/Multiple",
    "White":                            "White",
    "Other":                            "Other",
    "restaurant_cafe_canteen":          "Restaurant/Cafe/Canteen",
    "retailers___other":                "Other Retailers",
    "takeaway_sandwich_shop":           "Takeaway & Sandwich Shops",
    "pub_bar_nightclub":                "Pubs, Bars & Night Clubs",
    "other_catering_premises":          "Other Caterers",
    "supermarkets_hypermarkets":        "Supermarkets & Hypermarkets",
    "hotel_bed_and_breakfast_guest_house": "Hotels & Guesthouses",
    "Population_density":               "Population Density",
}
labels = [LABEL_MAP.get(c, c) for c in corr.columns]

n = len(corr)
fig, ax = plt.subplots(figsize=(n * 0.55 + 2, n * 0.55 + 2))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
ax.set_yticklabels(labels, fontsize=7)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                fontsize=5, color="black" if abs(corr.values[i, j]) < 0.7 else "white")
ax.set_title("Correlation Matrix",
             fontsize=12, pad=12)
plt.tight_layout()
plt.savefig("NBR/vif_final/8_full_correlation_heatmap.png", dpi=150)
plt.close()

print("结果已保存：")
print("  NBR/vif_final/8_full_correlation_matrix.csv")
print("  NBR/vif_final/8_full_correlation_heatmap.png")
