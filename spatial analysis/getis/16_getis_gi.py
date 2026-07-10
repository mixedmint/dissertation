import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from libpysal.weights import Queen
from esda.getisord import G_Local

# ── 读取数据 ──
agg = pd.read_csv("data processing/4_FHRS_agg_MSOA.csv", encoding="utf-8-sig")
msoa = gpd.read_file("0_raw/2021 London MSOA/2021_London_MSOA.shp")[["MSOA21CD", "MSOA21NM", "geometry"]]
gdf = msoa.merge(agg[["MSOA21CD", "low_rating_pct", "restaurant_count"]], on="MSOA21CD", how="left")
gdf["low_rating_pct"] = gdf["low_rating_pct"].fillna(0)

print(f"样本量：{len(gdf):,} 个 MSOA\n")

# ── 空间权重矩阵（Queen Contiguity，Binary）──
W = Queen.from_dataframe(gdf, idVariable="MSOA21CD")
W.transform = "b"
print(f"平均邻居数：{W.mean_neighbors:.2f}")

# ── Getis-Ord Gi* ──
gi = G_Local(gdf["low_rating_pct"].values, W, star=True, permutations=999)

gdf["Gi_z"]   = gi.Zs
gdf["Gi_p"]   = gi.p_sim

# 显著性分类（p < 0.05）
def classify(row):
    if row["Gi_p"] >= 0.05:
        return "Not Significant"
    if row["Gi_z"] > 0:
        return "Hot Spot"
    return "Cold Spot"

gdf["Gi_class"] = gdf.apply(classify, axis=1)

# ── 保存结果 ──
gdf.drop(columns="geometry").to_csv("spatial analysis/getis/16_getis_gi_results.csv",
                                     index=False, encoding="utf-8-sig")
print("\n── Gi* 分类汇总 ──")
print(gdf["Gi_class"].value_counts().to_string())

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
# 图1：Hot / Cold Spot 分类地图
# ══════════════════════════════════════════
color_map = {
    "Hot Spot":        "#d7191c",
    "Cold Spot":       "#2c7bb6",
    "Not Significant": "#d9d9d9",
}
n_hot  = (gdf["Gi_class"] == "Hot Spot").sum()
n_cold = (gdf["Gi_class"] == "Cold Spot").sum()
n_ns   = (gdf["Gi_class"] == "Not Significant").sum()

fig, ax = plt.subplots(figsize=(10, 8))
for cls, color in color_map.items():
    gdf[gdf["Gi_class"] == cls].plot(color=color, linewidth=0.2,
                                      edgecolor="white", ax=ax)

patches = [
    mpatches.Patch(color="#d7191c", label=f"Hot Spot (p < 0.05)    n = {n_hot}"),
    mpatches.Patch(color="#2c7bb6", label=f"Cold Spot (p < 0.05)   n = {n_cold}"),
    mpatches.Patch(color="#d9d9d9", label=f"Not Significant            n = {n_ns}"),
]
ax.legend(handles=patches, loc="lower right", fontsize=9,
          title="Getis-Ord Gi*", title_fontsize=9, framealpha=0.9)

ax.set_title("Spatial Clusters of Low-Rated Outlets",
             fontsize=14, fontweight="bold", pad=10)
ax.axis("off")
add_scalebar(ax)
add_north_arrow(ax)
plt.tight_layout()
plt.savefig("spatial analysis/getis/16_getis_gi_clusters.png", dpi=200, bbox_inches="tight")
plt.close()
print("\n图1已保存：getis/16_getis_gi_clusters.png")

# ══════════════════════════════════════════
# 图2：Gi* Z 值连续地图
# ══════════════════════════════════════════
from matplotlib.colors import TwoSlopeNorm

vmin, vmax = gdf["Gi_z"].min(), gdf["Gi_z"].max()
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

fig, ax = plt.subplots(figsize=(11, 8))
plt.subplots_adjust(right=0.88)
gdf.plot(column="Gi_z", cmap="RdBu_r", norm=norm,
         linewidth=0.2, edgecolor="grey", ax=ax)

cax = fig.add_axes([0.90, 0.15, 0.025, 0.45])
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("Gi* Z-score", fontsize=10, labelpad=8)
cbar.ax.tick_params(labelsize=8)

ax.set_title("Z-Score of Low-Rated Outlet Concentration",
             fontsize=14, fontweight="bold", pad=10)
ax.axis("off")
add_scalebar(ax)
add_north_arrow(ax)
plt.savefig("spatial analysis/getis/16_getis_gi_zscore.png", dpi=200, bbox_inches="tight")
plt.close()
print("图2已保存：getis/16_getis_gi_zscore.png")
print("\n结果已保存：getis/16_getis_gi_results.csv")
