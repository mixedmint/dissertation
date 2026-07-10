import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data processing/3_FHRS_MSOA.csv", encoding="utf-8-sig")

# 按 BusinessType 汇总
stats = (
    df.groupby("BusinessType")
    .agg(
        total=("RatingValue", "count"),
        low_count=("RatingValue", lambda x: (x <= 2).sum()),
    )
    .reset_index()
)
stats["high_count"] = stats["total"] - stats["low_count"]
stats["low_pct"]   = (stats["low_count"] / stats["total"] * 100).round(1)
stats = stats.sort_values("total", ascending=False)

stats.to_csv("spatial analysis/14_biztype_lowrating.csv", index=False, encoding="utf-8-sig")
print(stats.to_string(index=False))

# 堆叠柱状图
LABEL_MAP = {
    "Restaurant/Cafe/Canteen":               "Restaurant, Cafes\n& Canteens",
    "Retailers - other":                     "Other\nRetailers",
    "Takeaway/sandwich shop":                "Takeaways &\nSandwich Shops",
    "Pub/bar/nightclub":                     "Pubs, Bars &\nNightclubs",
    "Other catering premises":               "Other\nCaterers",
    "Retailers - supermarkets/hypermarkets": "Super &\nHypermarkets",
    "Hotel/bed & breakfast/guest house":     "Hotels,\nGuesthouses, B&B's",
}
stats["label"] = stats["BusinessType"].map(LABEL_MAP).fillna(stats["BusinessType"])

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("#f0f0f0")
ax.set_facecolor("#f0f0f0")

x = range(len(stats))
ax.bar(x, stats["high_count"], color="#3b6fbe", label="High-Rated (3–5)")
ax.bar(x, stats["low_count"],  color="#b03a2e", label="Low-Rated (0–2)",
       bottom=stats["high_count"])

# 百分比标注
for i, (_, row) in enumerate(stats.iterrows()):
    ax.text(i, row["total"] + stats["total"].max() * 0.01,
            f"{row['low_pct']}%", ha="center", va="bottom",
            fontsize=10, fontweight="bold")

ax.set_xticks(list(x))
ax.set_xticklabels(stats["label"], fontsize=10)
ax.set_ylabel("Number of Food Outlets", fontsize=11)
ax.set_title("Food Hygiene Rating by Business Type", fontsize=14, fontweight="bold", pad=14)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.legend(title="Rating", fontsize=10, title_fontsize=11,
          loc="upper right", framealpha=0.9)
ax.grid(axis="y", color="white", linewidth=0.8)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("spatial analysis/14_biztype_lowrating.png", dpi=200, bbox_inches="tight")
plt.close()
print("\n图已保存：spatial analysis/14_biztype_lowrating.png")
