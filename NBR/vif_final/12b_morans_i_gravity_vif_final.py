import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.formula.api as smf
from scipy.spatial.distance import cdist
from libpysal.weights.util import full2W
from esda.moran import Moran

# ── NBR（VIF筛选变量集）残差 Moran's I，用 gravity-like 距离权重替代 Queen 邻接权重 ──
# 权重定义：w_ij = 1 / d_ij^BETA（i≠j，d_ij 为 MSOA 质心间欧氏距离；自身权重为 0），
# 和 spatial analysis/getis/16b_morans_i_pct_gravity.py 保持同一套定义，便于跨模型对比。
BETA = 1

# ── 重跑 VIF 筛选变量集的 NBR，获取残差（与 12_morans_i_vif_final.py 完全一致）──
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
cols_needed = ["low_rating_count", "IMD", "log_restaurant_count", "MSOA21CD"] + controls
df_clean = df[cols_needed].dropna()
df_clean = df_clean[np.isfinite(df_clean.drop(columns="MSOA21CD")).all(axis=1)]

formula = "low_rating_count ~ IMD + " + " + ".join(controls)
model  = smf.negativebinomial(formula, data=df_clean,
                              offset=df_clean["log_restaurant_count"])
result = model.fit(disp=False)

mu = result.predict()
pearson_resid = (df_clean["low_rating_count"] - mu) / np.sqrt(mu)

# ── 建立 gravity 权重矩阵 ──
msoa = gpd.read_file("0_raw/2021 London MSOA/2021_London_MSOA.shp")[["MSOA21CD", "geometry"]]
msoa = msoa[msoa["MSOA21CD"].isin(df_clean["MSOA21CD"])].copy()
msoa = msoa.set_index("MSOA21CD").loc[df_clean["MSOA21CD"].values].reset_index()

coords = np.column_stack([msoa.geometry.centroid.x.values, msoa.geometry.centroid.y.values])
dist = cdist(coords, coords)
np.fill_diagonal(dist, np.inf)
weights = 1.0 / (dist ** BETA)

W = full2W(weights, ids=msoa["MSOA21CD"].tolist())
W.transform = "r"

# ── Moran's I ──
mi = Moran(pearson_resid.values, W)

print(f"Gravity 权重设定：w_ij = 1 / d_ij^{BETA}（行标准化）")
print("\n── NBR（VIF筛选变量集）残差 Moran's I（gravity 距离权重）──")
print(f"  Moran's I:  {mi.I:.4f}")
print(f"  期望值 E[I]: {mi.EI:.4f}")
print(f"  z-score:    {mi.z_norm:.4f}")
print(f"  p-value:    {mi.p_norm:.4f}")
print(f"  结论: {'存在显著空间自相关' if mi.p_norm < 0.05 else '无显著空间自相关'}")

# ── 与 Queen 邻接权重结果对比 ──
try:
    queen_result = pd.read_csv("NBR/vif_final/12_morans_i_vif_final.csv", encoding="utf-8-sig")
    print("\n── 与 Queen 邻接权重对比 ──")
    print(f"  Queen   Moran's I = {queen_result['Moran_I'].iloc[0]:.4f}  "
          f"(z={queen_result['z_score'].iloc[0]:.2f}, p={queen_result['p_value'].iloc[0]:.4f})")
    print(f"  Gravity Moran's I = {mi.I:.4f}  (z={mi.z_norm:.2f}, p={mi.p_norm:.4f})")
except FileNotFoundError:
    pass

out = pd.DataFrame([{
    "Model":       "NBR (vif_final)",
    "Weight_Type": f"Gravity (w_ij = 1/d_ij^{BETA})",
    "Moran_I":     round(mi.I, 4),
    "E_I":         round(mi.EI, 4),
    "z_score":     round(mi.z_norm, 4),
    "p_value":     round(mi.p_norm, 4),
}])
out.to_csv("NBR/vif_final/12b_morans_i_gravity_vif_final.csv", index=False, encoding="utf-8-sig")
print("\n结果已保存：NBR/vif_final/12b_morans_i_gravity_vif_final.csv")
