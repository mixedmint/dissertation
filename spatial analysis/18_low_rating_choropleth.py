import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon

# ── 读取数据 ──
agg  = pd.read_csv("data processing/4_FHRS_agg_MSOA.csv", encoding="utf-8-sig")
msoa = gpd.read_file("0_raw/2021 London MSOA/2021_London_MSOA.shp")[["MSOA21CD", "geometry"]]
gdf  = msoa.merge(agg[["MSOA21CD", "low_rating_pct"]], on="MSOA21CD", how="left")
gdf["low_rating_pct"] = gdf["low_rating_pct"].fillna(0)

# ── 分级 ──
# 28.8% 的 MSOA low_rating_pct 恰好为 0（无低分店），若直接five等分位数会导致
# 前几档全部挤在 0，因此单独给 0% 一档，再对非零部分做四等分位数
zero_mask = gdf["low_rating_pct"] == 0
nonzero = gdf.loc[~zero_mask, "low_rating_pct"]
q = nonzero.quantile([0, .25, .5, .75, 1.0]).values
q[-1] += 0.01

labels = [
    f"0%  (No Low-Rated Outlets, n={zero_mask.sum()})",
    f"0.0–{q[1]:.1f}%",
    f"{q[1]:.1f}–{q[2]:.1f}%",
    f"{q[2]:.1f}–{q[3]:.1f}%",
    f"{q[3]:.1f}–{q[4]:.1f}%  (Highest)",
]
colors = ["#feedde", "#fdbe85", "#fd8d3c", "#e6550d", "#a63603"]

gdf["pct_class"] = pd.NA
gdf.loc[zero_mask, "pct_class"] = labels[0]
nonzero_bins = pd.cut(nonzero, bins=q, labels=labels[1:], include_lowest=True)
gdf.loc[~zero_mask, "pct_class"] = nonzero_bins.astype(str)
gdf["pct_class"] = pd.Categorical(gdf["pct_class"], categories=labels, ordered=True)

# ══════════════════════════════════════════
# 比例尺 & 指北针
# ══════════════════════════════════════════
def add_scalebar(ax, length_m=20000, n_seg=4):
    xmin_, xmax_ = ax.get_xlim()
    ymin_, ymax_ = ax.get_ylim()
    px = (xmax_ - xmin_) * 0.04
    x0 = xmin_ + px
    y0 = ymin_ + (ymax_ - ymin_) * 0.04
    bar_h = (ymax_ - ymin_) * 0.012
    seg = length_m / n_seg
    for i in range(n_seg):
        ax.fill_between([x0 + i*seg, x0 + (i+1)*seg], [y0]*2, [y0+bar_h]*2,
                        color="black" if i % 2 == 0 else "white",
                        edgecolor="black", linewidth=0.5, zorder=5)
    for i in range(n_seg + 1):
        ax.text(x0 + i*seg, y0 - bar_h*0.6, str(int(i * length_m / n_seg / 1000)),
                ha="center", va="top", fontsize=7, zorder=5)
    ax.text(x0 + length_m + px*0.2, y0 + bar_h*0.5, "km",
            ha="left", va="center", fontsize=7, zorder=5)

def add_north_arrow(ax):
    xmin_, xmax_ = ax.get_xlim()
    ymin_, ymax_ = ax.get_ylim()
    cx = xmax_ - (xmax_ - xmin_) * 0.05
    cy = ymax_ - (ymax_ - ymin_) * 0.05
    h = (ymax_ - ymin_) * 0.055
    w = (xmax_ - xmin_) * 0.014
    upper = np.array([[cx, cy], [cx - w/2, cy - h/2], [cx + w/2, cy - h/2]])
    lower = np.array([[cx, cy - h], [cx - w/2, cy - h/2], [cx + w/2, cy - h/2]])
    ax.add_patch(MplPolygon(upper, closed=True, color="black", zorder=6))
    ax.add_patch(MplPolygon(lower, closed=True, facecolor="white",
                            edgecolor="black", linewidth=0.8, zorder=6))
    ax.text(cx, cy + h*0.1, "N", ha="center", va="bottom",
            fontsize=9, fontweight="bold", zorder=7)

# ══════════════════════════════════════════
# 绘图
# ══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))

for label, color in zip(labels, colors):
    subset = gdf[gdf["pct_class"] == label]
    if len(subset) > 0:
        subset.plot(color=color, linewidth=0.2, edgecolor="white", ax=ax)

legend_patches = [mpatches.Patch(color=c, label=l)
                  for c, l in zip(colors, labels)]
ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
          title="Low-Rated Outlets (%)", title_fontsize=9, framealpha=0.9)

ax.set_title("Proportion of Low-Rated Outlets by MSOA",
             fontsize=14, fontweight="bold", pad=10)

ax.axis("off")
add_scalebar(ax)
add_north_arrow(ax)
plt.tight_layout()
plt.savefig("spatial analysis/18_low_rating_choropleth.png", dpi=200, bbox_inches="tight")
plt.close()
print("完成！已保存：spatial analysis/18_low_rating_choropleth.png")
print(f"\n各分级 MSOA 数量：")
print(gdf["pct_class"].value_counts().sort_index().to_string())
