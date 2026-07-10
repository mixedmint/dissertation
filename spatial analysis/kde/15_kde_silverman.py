import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import warnings
warnings.filterwarnings("ignore")

# ── 读取点数据并投影到 BNG ──
def load_points(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:27700")
    return gdf["geometry"].x.values, gdf["geometry"].y.values

low_x,  low_y  = load_points("spatial analysis/13_low_rated.csv")
high_x, high_y = load_points("spatial analysis/13_high_rated.csv")

# ── 读取 London 边界（用于裁剪和底图）──
msoa = gpd.read_file("0_raw/2021 London MSOA/2021_London_MSOA.shp").to_crs("EPSG:27700")
london = msoa.dissolve()

# 网格范围（稍微超出边界）
xmin, ymin, xmax, ymax = msoa.total_bounds
pad = 2000
grid_x = np.linspace(xmin - pad, xmax + pad, 500)
grid_y = np.linspace(ymin - pad, ymax + pad, 500)
xx, yy = np.meshgrid(grid_x, grid_y)
grid_coords = np.vstack([xx.ravel(), yy.ravel()])

# ── 计算 KDE（标准化坐标后指定带宽，避免尺度问题）──
def compute_kde(x, y, grid):
    # 标准化到 [0,1] 空间
    x_range = xmax - xmin
    y_range = ymax - ymin
    xn = (x - xmin) / x_range
    yn = (y - ymin) / y_range
    gxn = (grid[0] - xmin) / x_range
    gyn = (grid[1] - ymin) / y_range
    kde = gaussian_kde(np.vstack([xn, yn]), bw_method="silverman")
    return kde(np.vstack([gxn, gyn])).reshape(xx.shape)

print("计算 Low-Rated KDE...")
z_low  = compute_kde(low_x,  low_y,  grid_coords)
print("计算 High-Rated KDE...")
z_high = compute_kde(high_x, high_y, grid_coords)

# ── 自定义配色：透明→颜色 ──
def make_cmap(color):
    return LinearSegmentedColormap.from_list(
        "custom", [(1, 1, 1, 0), color], N=256
    )

cmap_low  = make_cmap("#b03a2e")   # 红色系
cmap_high = make_cmap("#1a5276")   # 蓝色系

# ── 比例尺 & 指北针（与 GWNBR 图保持一致）──
def add_scalebar(ax, length_m=20000, n_seg=4):
    xmin_, xmax_ = ax.get_xlim()
    ymin_, ymax_ = ax.get_ylim()
    px = (xmax_ - xmin_) * 0.04
    py = (ymax_ - ymin_) * 0.04
    x0 = xmin_ + px
    y0 = ymin_ + py
    bar_h = (ymax_ - ymin_) * 0.012
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

def add_north_arrow(ax):
    xmin_, xmax_ = ax.get_xlim()
    ymin_, ymax_ = ax.get_ylim()
    px = (xmax_ - xmin_) * 0.05
    py = (ymax_ - ymin_) * 0.05
    cx = xmax_ - px
    cy = ymax_ - py
    h  = (ymax_ - ymin_) * 0.055
    w  = (xmax_ - xmin_) * 0.014
    upper = np.array([[cx, cy], [cx - w/2, cy - h/2], [cx + w/2, cy - h/2]])
    lower = np.array([[cx, cy - h], [cx - w/2, cy - h/2], [cx + w/2, cy - h/2]])
    ax.add_patch(MplPolygon(upper, closed=True, color="black", zorder=6))
    ax.add_patch(MplPolygon(lower, closed=True, facecolor="white", edgecolor="black",
                            linewidth=0.8, zorder=6))
    ax.text(cx, cy + h*0.1, "N", ha="center", va="bottom",
            fontsize=9, fontweight="bold", zorder=7)

# ── 绘图函数 ──
def plot_kde(z, cmap, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))

    # 底图
    msoa.plot(ax=ax, color="#e8e8e8", edgecolor="#cccccc", linewidth=0.15)

    # 等高线层级（跳过最低密度区，避免全图染色）
    vmax = np.percentile(z, 99)
    levels = np.linspace(np.percentile(z, 60), vmax, 20)

    cf = ax.contourf(xx, yy, z, levels=levels, cmap=cmap, alpha=0.85, extend="max")

    # 用伦敦边界裁剪（clip to London）
    clip_path = london.geometry.iloc[0]
    for col in cf.collections:
        col.set_clip_path(
            plt.matplotlib.patches.PathPatch(
                plt.matplotlib.path.Path.make_compound_path(
                    *[plt.matplotlib.path.Path(np.array(p.exterior.coords))
                      for p in (clip_path.geoms if hasattr(clip_path, "geoms") else [clip_path])]
                ),
                transform=ax.transData
            )
        )

    cbar = plt.colorbar(cf, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Kernel Density Estimate", fontsize=10)

    london.boundary.plot(ax=ax, color="#444444", linewidth=1.0)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")
    add_scalebar(ax)
    add_north_arrow(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"已保存：{out_path}")

plot_kde(z_low,  cmap_low,
         f"KDE Map of Low-Rated Outlets (Score 0–2)\nn = {len(low_x):,}",
         "spatial analysis/kde/15_kde_silverman_low.png")

plot_kde(z_high, cmap_high,
         f"KDE Map of High-Rated Outlets (Score 3–5)\nn = {len(high_x):,}",
         "spatial analysis/kde/15_kde_silverman_high.png")

print("完成！")
