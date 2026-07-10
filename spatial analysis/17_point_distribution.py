import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon

# ── 读取点数据并投影到 BNG ──
def load_gdf(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:27700")

low  = load_gdf("spatial analysis/13_low_rated.csv")
high = load_gdf("spatial analysis/13_high_rated.csv")

# ── 底图 ──
msoa   = gpd.read_file("0_raw/2021 London MSOA/2021_London_MSOA.shp").to_crs("EPSG:27700")
london = msoa.dissolve()

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
# 图1：合并图（高分蓝 + 低分红）
# ══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))
msoa.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.15)
london.boundary.plot(ax=ax, color="#555555", linewidth=1.0)

high.plot(ax=ax, color="#2166ac", markersize=0.8, alpha=0.3, zorder=2)
low.plot(ax=ax,  color="#d7191c", markersize=1.5, alpha=0.6, zorder=3)

patches = [
    mpatches.Patch(color="#2166ac", label=f"High-Rated (3–5)   n = {len(high):,}"),
    mpatches.Patch(color="#d7191c", label=f"Low-Rated (0–2)     n = {len(low):,}"),
]
ax.legend(handles=patches, loc="lower right", fontsize=9,
          title="FHRS Rating", title_fontsize=9, framealpha=0.9)

ax.set_title("Spatial Distribution of Food Outlets by Rating",
             fontsize=14, fontweight="bold", pad=10)
ax.axis("off")
add_scalebar(ax)
add_north_arrow(ax)
plt.tight_layout()
plt.savefig("spatial analysis/17_point_combined.png", dpi=200, bbox_inches="tight")
plt.close()
print("图1已保存：spatial analysis/17_point_combined.png")

# ══════════════════════════════════════════
# 图2：低分店单独分布图
# ══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))
msoa.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.15)
london.boundary.plot(ax=ax, color="#555555", linewidth=1.0)
low.plot(ax=ax, color="#d7191c", markersize=2, alpha=0.5, zorder=3)

ax.set_title(f"Spatial Distribution of Low-Rated Outlets (Score 0–2)\nn = {len(low):,}",
             fontsize=14, fontweight="bold", pad=10)
ax.axis("off")
add_scalebar(ax)
add_north_arrow(ax)
plt.tight_layout()
plt.savefig("spatial analysis/17_point_low.png", dpi=200, bbox_inches="tight")
plt.close()
print("图2已保存：spatial analysis/17_point_low.png")

# ══════════════════════════════════════════
# 图3：高分店单独分布图
# ══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))
msoa.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.15)
london.boundary.plot(ax=ax, color="#555555", linewidth=1.0)
high.plot(ax=ax, color="#2166ac", markersize=0.8, alpha=0.3, zorder=3)

ax.set_title(f"Spatial Distribution of High-Rated Outlets (Score 3–5)\nn = {len(high):,}",
             fontsize=14, fontweight="bold", pad=10)
ax.axis("off")
add_scalebar(ax)
add_north_arrow(ax)
plt.tight_layout()
plt.savefig("spatial analysis/17_point_high.png", dpi=200, bbox_inches="tight")
plt.close()
print("图3已保存：spatial analysis/17_point_high.png")
