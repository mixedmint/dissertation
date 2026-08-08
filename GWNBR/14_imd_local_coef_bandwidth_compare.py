import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon as MplPolygon

# ── 把三个带宽（140/186/240）的 Local IMD Coefficient 图拼成一张（上二下一）──
BANDWIDTH_DIRS = {
    140: "GWNBR_noblack/bandwidth=140",
    186: "GWNBR_noblack",
    240: "GWNBR_noblack/bandwidth=240",
}

msoa_geo = gpd.read_file("0_raw/2021 London MSOA/2021_London_MSOA.shp")[["MSOA21CD", "geometry"]]

gdfs = {}
for bw, d in BANDWIDTH_DIRS.items():
    coefs = pd.read_csv(f"{d}/11_gwnbr_local_coefs.csv")[["MSOA21CD", "IMD"]]
    gdf = msoa_geo.merge(coefs, on="MSOA21CD")
    gdfs[bw] = gdf

# 三个带宽共用同一套色阶，才能直接比较颜色深浅
all_vals = pd.concat([g["IMD"] for g in gdfs.values()])
vmin, vmax = all_vals.min(), all_vals.max()
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
cmap = "RdBu_r"

# ══════════════════════════════
# 比例尺 & 指北针
# ══════════════════════════════
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
                ha="center", va="top", fontsize=6, zorder=5)
    ax.text(x0 + length_m + px*0.2, y0 + bar_h*0.5, "km",
            ha="left", va="center", fontsize=6, zorder=5)

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
            fontsize=8, fontweight="bold", zorder=7)

# ══════════════════════════════
# 绘图：一行三列
# ══════════════════════════════
fig, ax_list = plt.subplots(1, 3, figsize=(19, 6.2))
axes = {140: ax_list[0], 186: ax_list[1], 240: ax_list[2]}

for bw, ax in axes.items():
    gdf = gdfs[bw]
    gdf.plot(column="IMD", cmap=cmap, norm=norm, linewidth=0.2, edgecolor="grey", ax=ax)
    tag = "  (Main Model)" if bw == 186 else ""
    ax.set_title(f"Bandwidth = {bw}{tag}", fontsize=13, fontweight="bold", pad=8)
    ax.axis("off")
    # 三张图比例尺、方向完全一致，分别标一次在最左（比例尺）和最右（指北针）即可
    if bw == 140:
        add_scalebar(ax)
    if bw == 240:
        add_north_arrow(ax)

fig.subplots_adjust(top=0.86, bottom=0.14, wspace=0.05)

# 共用色条
cax = fig.add_axes([0.32, 0.06, 0.36, 0.03])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
cbar.set_label("Local IMD Coefficient", fontsize=10, labelpad=6)
cbar.ax.tick_params(labelsize=8)

fig.suptitle("Spatial Variation in Local IMD Coefficient Across Bandwidths",
             fontsize=16, fontweight="bold", y=0.96)

plt.savefig("GWNBR_noblack/14_imd_local_coef_bandwidth_compare.png", dpi=200, bbox_inches="tight")
plt.close()
print("完成！已保存：GWNBR_noblack/14_imd_local_coef_bandwidth_compare.png")
