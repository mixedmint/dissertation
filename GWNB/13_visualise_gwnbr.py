import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon

# ── 读取数据 ──
coefs = pd.read_csv("GWNB/11_gwnbr_local_coefs.csv")
se    = pd.read_csv("GWNB/11_gwnbr_local_se.csv")

merged = coefs[["MSOA21CD", "IMD"]].merge(
    se[["MSOA21CD", "IMD"]].rename(columns={"IMD": "IMD_se"}), on="MSOA21CD"
)
merged["IMD_t"]       = merged["IMD"] / merged["IMD_se"]
merged["significant"] = merged["IMD_t"].abs() > 1.96

msoa = gpd.read_file("0_raw/2021 London MSOA/2021_London_MSOA.shp")[["MSOA21CD", "geometry"]]
gdf  = msoa.merge(merged, on="MSOA21CD")
n_sig = int(gdf["significant"].sum())

# ══════════════════════════════
# 辅助函数
# ══════════════════════════════
def add_scalebar(ax, length_m=20000, n_seg=4):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    px = (xmax - xmin) * 0.04
    py = (ymax - ymin) * 0.04
    x0 = xmin + px
    y0 = ymin + py
    bar_h = (ymax - ymin) * 0.012
    seg   = length_m / n_seg

    for i in range(n_seg):
        color = "black" if i % 2 == 0 else "white"
        ax.fill_between([x0 + i*seg, x0 + (i+1)*seg], [y0]*2, [y0+bar_h]*2,
                        color=color, edgecolor="black", linewidth=0.5, zorder=5)
    for i in range(n_seg + 1):
        km = int(i * length_m / n_seg / 1000)
        ax.text(x0 + i*seg, y0 - bar_h*0.6, str(km),
                ha="center", va="top", fontsize=7, zorder=5)
    ax.text(x0 + length_m + px*0.2, y0 + bar_h*0.5, "km",
            ha="left", va="center", fontsize=7, zorder=5)

def add_north_arrow(ax, loc="upper right"):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    px = (xmax - xmin) * 0.05
    py = (ymax - ymin) * 0.05
    cx = xmax - px
    cy = ymax - py
    h  = (ymax - ymin) * 0.055
    w  = (xmax - xmin) * 0.014

    upper = np.array([[cx, cy], [cx - w/2, cy - h/2], [cx + w/2, cy - h/2]])
    lower = np.array([[cx, cy - h], [cx - w/2, cy - h/2], [cx + w/2, cy - h/2]])
    ax.add_patch(Polygon(upper, closed=True, color="black", zorder=6))
    ax.add_patch(Polygon(lower, closed=True, facecolor="white", edgecolor="black",
                         linewidth=0.8, zorder=6))
    ax.text(cx, cy + h*0.1, "N", ha="center", va="bottom",
            fontsize=9, fontweight="bold", zorder=7)

# ════════════════════════════════════════════════
# 图1：红蓝渐变，色条在左侧
# ════════════════════════════════════════════════
vmin = gdf["IMD"].min()
vmax = gdf["IMD"].max()
norm1 = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

fig, ax = plt.subplots(figsize=(11, 8))
plt.subplots_adjust(right=0.88)

gdf.plot(column="IMD", cmap="RdBu_r", norm=norm1,
         linewidth=0.2, edgecolor="grey", ax=ax)

# 色条放右下角
cax = fig.add_axes([0.86, 0.12, 0.025, 0.45])
sm1 = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm1)
sm1.set_array([])
cbar = fig.colorbar(sm1, cax=cax)
cbar.set_label("Local IMD Coefficient", fontsize=10, labelpad=8)
cbar.ax.tick_params(labelsize=8)

ax.set_title("Spatial Variation in IMD Coefficient", fontsize=15, fontweight="bold", pad=28)
ax.text(0.5, 1.02, "Bandwidth = 186,  n = 1,002 MSOAs", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=11, color="#444444")
ax.axis("off")
add_scalebar(ax)
add_north_arrow(ax)
plt.savefig("GWNB/13_imd_local_coef.png", dpi=200, bbox_inches="tight")
plt.close()
print("图1已保存：GWNB/13_imd_local_coef.png")

# ════════════════════════════════════════════════
# 图2：显著性（二分类）
# ════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))

gdf[~gdf["significant"]].plot(color="#d9d9d9", linewidth=0.2, edgecolor="grey", ax=ax)
gdf[gdf["significant"]].plot(color="#2166ac",  linewidth=0.2, edgecolor="grey", ax=ax)

patch_sig   = mpatches.Patch(color="#2166ac", label=f"Significant (|t| > 1.96)   n = {n_sig}")
patch_insig = mpatches.Patch(color="#d9d9d9", label=f"Not Significant                n = {len(gdf)-n_sig}")
legend = ax.legend(handles=[patch_sig, patch_insig], loc="lower right", fontsize=9,
                   title="IMD Coefficient Significance", title_fontsize=9, framealpha=0.9)

ax.set_title("Significance of Local IMD Coefficient", fontsize=15, fontweight="bold", pad=28)
ax.text(0.5, 1.02, "308 / 1002 MSOAs significant at 95% confidence level",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=11, color="#444444")
ax.axis("off")
add_scalebar(ax)
add_north_arrow(ax)
plt.tight_layout()
plt.savefig("GWNB/13_imd_significance.png", dpi=200, bbox_inches="tight")
plt.close()
print("图2已保存：GWNB/13_imd_significance.png")

# ════════════════════════════════════════════════
# 图3：显著区域系数强度（分级蓝色）
# ════════════════════════════════════════════════
bins3   = [-0.30, -0.25, -0.20, -0.15, -0.10, -0.05]
labels3 = ["-0.30 to -0.25", "-0.25 to -0.20", "-0.20 to -0.15",
           "-0.15 to -0.10", "-0.10 to -0.05"]
colors3 = ["#08306b", "#2171b5", "#6baed6", "#bdd7e7", "#eff3ff"]

gdf["IMD_class3"] = pd.cut(gdf["IMD"], bins=bins3, labels=labels3, right=False)

fig, ax = plt.subplots(figsize=(10, 8))

gdf[~gdf["significant"]].plot(color="#d9d9d9", linewidth=0.2, edgecolor="grey", ax=ax)
for label, color in zip(labels3, colors3):
    subset = gdf[gdf["significant"] & (gdf["IMD_class3"] == label)]
    if len(subset) > 0:
        subset.plot(color=color, linewidth=0.2, edgecolor="grey", ax=ax)

legend_patches3 = [mpatches.Patch(color=c, label=l) for c, l in zip(colors3, labels3)]
legend_patches3.append(mpatches.Patch(color="#d9d9d9", label="Not Significant (|t| ≤ 1.96)"))
leg3 = ax.legend(handles=legend_patches3, loc="lower right", fontsize=9,
                 title="Local IMD Coefficient", title_fontsize=9, framealpha=0.9)

ax.set_title("IMD Coefficient in Significant Areas", fontsize=15, fontweight="bold", pad=28)
ax.text(0.5, 1.02, f"{n_sig} / {len(gdf)} MSOAs significant at 95% confidence level",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=11, color="#444444")
ax.axis("off")
add_scalebar(ax)
add_north_arrow(ax)
plt.tight_layout()
plt.savefig("GWNB/13_imd_coef_significant_only.png", dpi=200, bbox_inches="tight")
plt.close()
print("图3已保存：GWNB/13_imd_coef_significant_only.png")
