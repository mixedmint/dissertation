import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.patches as mpatches
from libpysal.weights import Queen
from esda.moran import Moran

# ══════════════════════════════════════════════════════
# 0. 共用函数：比例尺 & 指北针
# ══════════════════════════════════════════════════════
def add_scalebar(ax, length_m=20000, n_seg=4):
    xmin_, xmax_ = ax.get_xlim()
    ymin_, ymax_ = ax.get_ylim()
    x0    = xmin_ + (xmax_ - xmin_) * 0.04
    y0    = ymin_ + (ymax_ - ymin_) * 0.04
    bar_h = (ymax_ - ymin_) * 0.012
    seg   = length_m / n_seg
    for i in range(n_seg):
        ax.fill_between([x0 + i*seg, x0 + (i+1)*seg], [y0]*2, [y0+bar_h]*2,
                        color="black" if i % 2 == 0 else "white",
                        edgecolor="black", linewidth=0.5, zorder=5)
    for i in range(n_seg + 1):
        ax.text(x0 + i*seg, y0 - bar_h*0.6,
                str(int(i * length_m / n_seg / 1000)),
                ha="center", va="top", fontsize=6, zorder=5)
    ax.text(x0 + length_m + (xmax_ - xmin_)*0.008, y0 + bar_h*0.5,
            "km", ha="left", va="center", fontsize=6, zorder=5)

def add_north_arrow(ax):
    xmin_, xmax_ = ax.get_xlim()
    ymin_, ymax_ = ax.get_ylim()
    cx = xmax_ - (xmax_ - xmin_) * 0.05
    cy = ymax_ - (ymax_ - ymin_) * 0.06
    h  = (ymax_ - ymin_) * 0.055
    w  = (xmax_ - xmin_) * 0.014
    upper = np.array([[cx, cy], [cx - w/2, cy - h/2], [cx + w/2, cy - h/2]])
    lower = np.array([[cx, cy - h], [cx - w/2, cy - h/2], [cx + w/2, cy - h/2]])
    ax.add_patch(MplPolygon(upper, closed=True, color="black", zorder=6))
    ax.add_patch(MplPolygon(lower, closed=True, facecolor="white",
                            edgecolor="black", linewidth=0.8, zorder=6))
    ax.text(cx, cy + h*0.1, "N", ha="center", va="bottom",
            fontsize=8, fontweight="bold", zorder=7)

# ══════════════════════════════════════════════════════
# 1. 计算 NBR Pearson 残差
# ══════════════════════════════════════════════════════
df = pd.read_csv("data processing/7_regression.csv", encoding="utf-8-sig")
df["log_restaurant_count"] = np.log(df["restaurant_count"])
df["log_pop_density"]      = np.log(df["Population_density"])

controls = [
    "Asian", "Other",
    "retailers___other", "takeaway_sandwich_shop",
    "pub_bar_nightclub", "supermarkets_hypermarkets",
    "hotel_bed_and_breakfast_guest_house", "other_catering_premises",
    "log_pop_density",
]
cols = ["low_rating_count", "IMD", "log_restaurant_count", "MSOA21CD"] + controls
df_nbr = df[cols].dropna()
df_nbr = df_nbr[np.isfinite(df_nbr.drop(columns="MSOA21CD")).all(axis=1)]

formula = "low_rating_count ~ IMD + " + " + ".join(controls)
nbr_res = smf.negativebinomial(formula, data=df_nbr,
                               offset=df_nbr["log_restaurant_count"]).fit(disp=False)
mu_nbr  = nbr_res.predict()
resid_nbr = (df_nbr["low_rating_count"].values - mu_nbr) / np.sqrt(mu_nbr)

df_nbr = df_nbr.copy()
df_nbr["pearson_resid"] = resid_nbr
print(f"NBR（无 Black）残差  均值={resid_nbr.mean():.4f}  SD={resid_nbr.std():.4f}")

# 现算 NBR（无 Black）残差的 Moran's I，不复用主模型（含 Black）缓存的数值，
# 避免图上标题和实际画的残差对不上
msoa_w = gpd.read_file("0_raw/2021 London MSOA/2021_London_MSOA.shp")[["MSOA21CD", "geometry"]]
msoa_w = msoa_w[msoa_w["MSOA21CD"].isin(df_nbr["MSOA21CD"])].copy()
msoa_w = msoa_w.set_index("MSOA21CD").loc[df_nbr["MSOA21CD"].values].reset_index()
W_nbr = Queen.from_dataframe(msoa_w, idVariable="MSOA21CD")
W_nbr.transform = "r"
nbr_mi_local = Moran(resid_nbr, W_nbr)
print(f"NBR（无 Black）残差 Moran's I = {nbr_mi_local.I:.4f}  p={nbr_mi_local.p_norm:.4f}")

# ══════════════════════════════════════════════════════
# 2. 计算 GWNBR Pearson 残差（局部系数 × 原始数据）
# ══════════════════════════════════════════════════════
local = pd.read_csv("GWNBR/11_gwnbr_local_coefs.csv", encoding="utf-8-sig")

df_gw = pd.read_csv("GWNBR/10_regression_with_coords.csv", encoding="utf-8-sig")
df_gw["log_restaurant_count"] = np.log(df_gw["restaurant_count"])
df_gw["log_pop_density"]      = np.log(df_gw["Population_density"])

gw_vars = ["MSOA21CD", "low_rating_count", "log_restaurant_count",
           "IMD", "Asian", "Other",
           "retailers___other", "takeaway_sandwich_shop",
           "pub_bar_nightclub", "supermarkets_hypermarkets",
           "hotel_bed_and_breakfast_guest_house", "other_catering_premises",
           "log_pop_density"]
df_gw = df_gw[gw_vars].dropna()
df_gw = df_gw[np.isfinite(df_gw.drop(columns="MSOA21CD")).all(axis=1)].reset_index(drop=True)

x_vars = ["IMD", "Asian", "Other",
          "retailers___other", "takeaway_sandwich_shop",
          "pub_bar_nightclub", "supermarkets_hypermarkets",
          "hotel_bed_and_breakfast_guest_house", "other_catering_premises",
          "log_pop_density"]

merged = df_gw.merge(
    local[["MSOA21CD", "Intercept"] + x_vars],
    on="MSOA21CD", how="inner", suffixes=("", "_coef")
)

eta = merged["Intercept"].values.copy()
for v in x_vars:
    eta += merged[v].values * merged[f"{v}_coef"].values
eta += merged["log_restaurant_count"].values
mu_gw = np.exp(eta)

y_gw = merged["low_rating_count"].values
resid_gw = (y_gw - mu_gw) / np.sqrt(mu_gw)
merged["pearson_resid"] = resid_gw
print(f"GWNBR 残差  均值={resid_gw.mean():.4f}  SD={resid_gw.std():.4f}")

# ══════════════════════════════════════════════════════
# 3. 合并到 GeoDataFrame
# ══════════════════════════════════════════════════════
msoa = gpd.read_file("0_raw/2021 London MSOA/2021_London_MSOA.shp")[["MSOA21CD", "geometry"]]

gdf_nbr = msoa.merge(df_nbr[["MSOA21CD", "pearson_resid"]].rename(
                         columns={"pearson_resid": "resid_nbr"}),
                     on="MSOA21CD", how="left")

gdf_gw  = msoa.merge(merged[["MSOA21CD", "pearson_resid"]].rename(
                         columns={"pearson_resid": "resid_gw"}),
                     on="MSOA21CD", how="left")

gdf = gdf_nbr.merge(gdf_gw[["MSOA21CD", "resid_gw"]], on="MSOA21CD", how="left")
gdf["abs_resid_gw"] = gdf["resid_gw"].abs()

# ── Moran's I：NBR 用本脚本刚现算的（无 Black），GWNBR 读 17 脚本算好的（无 Black）──
def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.4f}"

gwnbr_mi = pd.read_csv("GWNBR/17_gwnbr_morans_i.csv", encoding="utf-8-sig").iloc[0]

# ══════════════════════════════════════════════════════
# 图1：NBR vs GWNBR 残差并排（共享色阶）
# ══════════════════════════════════════════════════════
# 用两组残差中的最大绝对值定色阶，保证对称
clip = np.nanpercentile(
    np.abs(np.concatenate([gdf["resid_nbr"].dropna(), gdf["resid_gw"].dropna()])), 97
)
norm = TwoSlopeNorm(vmin=-clip, vcenter=0, vmax=clip)
cmap = "RdBu_r"

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.subplots_adjust(wspace=0.08, right=0.88)

titles = [
    f"NBR Residuals \n(Moran's I = {nbr_mi_local.I:.4f}, {fmt_p(nbr_mi_local.p_norm)})",
    f"GWNBR Residuals \n(Moran's I = {gwnbr_mi['Moran_I']:.4f}, {fmt_p(gwnbr_mi['p_value'])})",
]
cols_r = ["resid_nbr", "resid_gw"]

for ax, col, title in zip(axes, cols_r, titles):
    gdf.plot(column=col, cmap=cmap, norm=norm,
             linewidth=0.15, edgecolor="grey", ax=ax, missing_kwds={"color": "#eeeeee"})
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.axis("off")
    add_scalebar(ax)
    add_north_arrow(ax)

# 共享色条
cax = fig.add_axes([0.90, 0.18, 0.022, 0.52])
sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("Pearson Residual", fontsize=10, labelpad=8)
cbar.ax.tick_params(labelsize=8)

fig.suptitle("Pearson Residuals of Global NBR Model and GWNBR Model",
             fontsize=14, fontweight="bold", y=0.97)

plt.savefig("GWNBR/18_residual_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("图1已保存：GWNBR/18_residual_comparison.png")

# ══════════════════════════════════════════════════════
# 图2：GWNBR 局部拟合优度（|残差|，越小越好）
# ══════════════════════════════════════════════════════
abs_max = np.nanpercentile(gdf["abs_resid_gw"], 97)
norm2   = Normalize(vmin=0, vmax=abs_max)

fig2, ax2 = plt.subplots(figsize=(10, 8))
gdf.plot(column="abs_resid_gw", cmap="YlOrRd", norm=norm2,
         linewidth=0.15, edgecolor="grey", ax=ax2,
         missing_kwds={"color": "#eeeeee"})

cax2 = fig2.add_axes([0.88, 0.18, 0.025, 0.52])
sm2  = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm2)
sm2.set_array([])
cbar2 = fig2.colorbar(sm2, cax=cax2)
cbar2.set_label("|Pearson Residual|", fontsize=10, labelpad=8)
cbar2.ax.tick_params(labelsize=8)

# 手动图例
legend_patches = [
    mpatches.Patch(color="#ffffb2", label="Low (Good Fit)"),
    mpatches.Patch(color="#fd8d3c", label="Medium"),
    mpatches.Patch(color="#bd0026", label="High (Poor Fit)"),
]
ax2.legend(handles=legend_patches, loc="lower right", fontsize=9,
           title="Local Fit Quality", title_fontsize=9, framealpha=0.9)

ax2.set_title("GWNBR Local Model Fit Quality\n(Absolute Pearson Residuals — Lower = Better Fit)",
              fontsize=13, fontweight="bold", pad=12)
ax2.axis("off")
add_scalebar(ax2)
add_north_arrow(ax2)
plt.tight_layout()
plt.savefig("GWNBR/18_local_fit_quality.png", dpi=200, bbox_inches="tight")
plt.close()
print("图2已保存：GWNBR/18_local_fit_quality.png")
