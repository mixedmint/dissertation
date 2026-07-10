import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("NBR/no_age/9_nbr_results_no_age.csv", encoding="utf-8-sig")

# 排除 Intercept 和 alpha
df = df[~df["Variable"].isin(["Intercept", "alpha"])].copy()

# IRR 的置信区间（exp(CI_coef)）
df["IRR_lower"] = np.exp(df["CI_lower"])
df["IRR_upper"] = np.exp(df["CI_upper"])

# 变量标签
LABEL_MAP = {
    "IMD":                              "IMD Decile",
    "Asian":                            "Asian (%)",
    "Black":                            "Black (%)",
    "Other":                            "Other Ethnicity (%)",
    "retailers___other":                "Other Retailers (%)",
    "takeaway_sandwich_shop":           "Takeaway & Sandwich (%)",
    "pub_bar_nightclub":                "Pubs, Bars & Nightclubs (%)",
    "supermarkets_hypermarkets":        "Supermarkets (%)",
    "hotel_bed_and_breakfast_guest_house": "Hotels & Guesthouses (%)",
    "other_catering_premises":          "Other Catering (%)",
    "log_pop_density":                  "Log Population Density",
}
df["label"] = df["Variable"].map(LABEL_MAP).fillna(df["Variable"])

# 显著性颜色
def dot_color(sig):
    if sig == "***": return "#d7191c"
    if sig == "**":  return "#e8711a"
    if sig == "*":   return "#f0a500"
    return "#888888"

df["color"] = df["Sig"].fillna("").apply(dot_color)

# 反序（从上到下读）
df = df.iloc[::-1].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9, 7))

y = np.arange(len(df))

# 误差线
for i, row in df.iterrows():
    ax.plot([row["IRR_lower"], row["IRR_upper"]], [i, i],
            color=row["color"], linewidth=1.2, zorder=2)

# 点
ax.scatter(df["IRR"], y, color=df["color"], s=55, zorder=3)

# IRR=1 参考线
ax.axvline(x=1, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

# 标签
ax.set_yticks(y)
ax.set_yticklabels(df["label"], fontsize=10)

# IRR 数值标注
for i, row in df.iterrows():
    sig = row["Sig"] if pd.notna(row["Sig"]) else ""
    ax.text(row["IRR_upper"] + 0.002, i,
            f'{row["IRR"]:.3f}{sig}',
            va="center", ha="left", fontsize=8.5, color=row["color"])

ax.set_xlabel("Incidence Rate Ratio (IRR)", fontsize=11)
ax.set_title("Negative Binomial Regression: IRR and 95% Confidence Intervals",
             fontsize=13, fontweight="bold", pad=12)

# 图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#d7191c", markersize=8, label="p < 0.001 (***)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#e8711a", markersize=8, label="p < 0.01  (**)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#f0a500", markersize=8, label="p < 0.05  (*)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888", markersize=8, label="Not Significant"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)

ax.set_xlim(left=df["IRR_lower"].min() - 0.01)
ax.grid(axis="x", linestyle=":", alpha=0.4)
plt.tight_layout()
plt.savefig("NBR/no_age/forest_plot.png", dpi=200, bbox_inches="tight")
plt.close()
print("已保存：NBR/no_age/forest_plot.png")
