import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data processing/4_FHRS_agg_MSOA.csv", encoding="utf-8-sig")

total    = len(df)
n_zero   = (df["low_rating_pct"] == 0).sum()
pct_zero = n_zero / total * 100

print(f"MSOA 总计：{total}")
print(f"low_rating_pct = 0：n = {n_zero}，占比 = {pct_zero:.1f}%")
print(f"low_rating_pct 描述（含零）：")
print(df["low_rating_pct"].describe().round(2).to_string())

df_nonzero = df[df["low_rating_pct"] > 0]
print(f"\nlow_rating_pct 描述（去零后，n={len(df_nonzero)}）：")
print(df_nonzero["low_rating_pct"].describe().round(2).to_string())

# ── 绘图 ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Distribution of Low-Rated Outlets (MSOA Level)",
             fontsize=14, fontweight="bold", y=1.01)

# 图1：含零分布
axes[0].hist(df["low_rating_pct"], bins=30, color="#3b6fbe", edgecolor="white", linewidth=0.4)
axes[0].set_xlabel("low_rating_pct (%)", fontsize=11)
axes[0].set_ylabel("Count", fontsize=11)
axes[0].set_title(f"Distribution of low_rating_pct\n(incl. zeros, n = {total:,})", fontsize=11)
axes[0].text(0.97, 0.97, f"{pct_zero:.1f}% = 0",
             transform=axes[0].transAxes, ha="right", va="top",
             fontsize=10, color="#b03a2e",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# 图2：去零分布
n_nz = len(df_nonzero)
axes[1].hist(df_nonzero["low_rating_pct"], bins=30, color="#e8711a", edgecolor="white", linewidth=0.4)
axes[1].set_xlabel("low_rating_pct (%)", fontsize=11)
axes[1].set_ylabel("Count", fontsize=11)
axes[1].set_title(f"Distribution (excl. zeros)\n(n = {n_nz:,})", fontsize=11)

# 图3：restaurant_count 分布
axes[2].hist(df["restaurant_count"], bins=30, color="#2e7d5e", edgecolor="white", linewidth=0.4)
axes[2].set_xlabel("restaurant_count", fontsize=11)
axes[2].set_ylabel("Count", fontsize=11)
axes[2].set_title(f"Restaurant Count per MSOA\n(n = {total:,})", fontsize=11)

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("data processing/low_rating_dist_msoa.png", dpi=200, bbox_inches="tight")
plt.close()
print("\n已保存：data processing/low_rating_dist_msoa.png")
