import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon

# ── 读取数据 ──
imd  = pd.read_csv("data processing/2b_IMD_Score_MSOA_London.csv", encoding="utf-8-sig")
msoa = gpd.read_file("0_raw/2021 London MSOA/2021_London_MSOA.shp")[["MSOA21CD", "geometry"]]
gdf  = msoa.merge(imd, on="MSOA21CD", how="left")

# ── 分级（5 分位数；IMD Score 越高＝越贫困）──
edges = gdf["IMD_Score_PopWeighted"].quantile([0, .2, .4, .6, .8, 1.0]).values
edges[0]  -= 0.01
edges[-1] += 0.01

labels = []
for i in range(5):
    lo, hi = edges[i], edges[i + 1]
    tag = "  (Least Deprived)" if i == 0 else "  (Most Deprived)" if i == 4 else ""
    labels.append(f"{lo:.1f}–{hi:.1f}{tag}")
colors = ["#eff3ff", "#bdd7e7", "#6baed6", "#2171b5", "#084594"]

gdf["IMD_class"] = pd.cut(gdf["IMD_Score_PopWeighted"], bins=edges,
                           labels=labels, include_lowest=True)

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
    subset = gdf[gdf["IMD_class"] == label]
    if len(subset) > 0:
        subset.plot(color=color, linewidth=0.2, edgecolor="white", ax=ax)

legend_patches = [mpatches.Patch(color=c, label=l)
                  for c, l in zip(colors, labels)]
ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
          title="IMD Score", title_fontsize=9, framealpha=0.9)

ax.set_title("IMD Score by MSOA",
             fontsize=14, fontweight="bold", pad=10)

ax.axis("off")
add_scalebar(ax)
add_north_arrow(ax)
plt.tight_layout()
plt.savefig("data processing/IMD_map.png", dpi=200, bbox_inches="tight")
plt.close()
print("完成！已保存：data processing/IMD_map.png")
print(f"\n各分级 MSOA 数量：")
print(gdf["IMD_class"].value_counts().sort_index().to_string())
