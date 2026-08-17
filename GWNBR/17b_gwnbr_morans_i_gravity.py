import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial.distance import cdist
from libpysal.weights.util import full2W
from esda.moran import Moran

# ── GWNBR 残差 Moran's I，用 gravity-like 距离权重替代 Queen 邻接权重 ──
# 权重定义与 12b_morans_i_gravity_vif_final.py / 16b_morans_i_pct_gravity.py 一致：
# w_ij = 1 / d_ij^BETA（i≠j，d_ij 为 MSOA 质心间欧氏距离；自身权重为 0）
BETA = 1

# ── 1. 读取局部系数 ──
local = pd.read_csv("GWNBR/11_gwnbr_local_coefs.csv", encoding="utf-8-sig")

# ── 2. 读取原始数据（与 17_gwnbr_morans_i.py 相同的预处理）──
df = pd.read_csv("GWNBR/10_regression_with_coords.csv", encoding="utf-8-sig")
df = df.rename(columns={"0-4": "age_0_4", "65plus": "age_65plus"})
df["log_restaurant_count"] = np.log(df["restaurant_count"])
df["log_pop_density"]      = np.log(df["Population_density"])

vars_needed = [
    "MSOA21CD", "low_rating_count", "log_restaurant_count",
    "IMD", "Asian", "Other",
    "retailers___other", "takeaway_sandwich_shop",
    "pub_bar_nightclub", "supermarkets_hypermarkets",
    "hotel_bed_and_breakfast_guest_house", "other_catering_premises",
    "log_pop_density"
]
df = df[vars_needed].dropna()
df = df[np.isfinite(df.drop(columns="MSOA21CD")).all(axis=1)].reset_index(drop=True)

# ── 3. 按 MSOA21CD 合并局部系数 ──
merged = df.merge(local[["MSOA21CD",
                          "Intercept", "IMD", "Asian", "Other",
                          "retailers___other", "takeaway_sandwich_shop",
                          "pub_bar_nightclub", "supermarkets_hypermarkets",
                          "hotel_bed_and_breakfast_guest_house", "other_catering_premises",
                          "log_pop_density"]],
                  on="MSOA21CD", how="inner", suffixes=("", "_coef"))

print(f"匹配样本量：{len(merged)}")

# ── 4. 计算 GWNBR 拟合值（局部线性预测量 + offset）──
x_vars = ["IMD", "Asian", "Other",
          "retailers___other", "takeaway_sandwich_shop",
          "pub_bar_nightclub", "supermarkets_hypermarkets",
          "hotel_bed_and_breakfast_guest_house", "other_catering_premises",
          "log_pop_density"]

eta = merged["Intercept"].values.copy()
for v in x_vars:
    eta += merged[v].values * merged[f"{v}_coef"].values
eta += merged["log_restaurant_count"].values   # offset

mu = np.exp(eta)

# ── 5. Pearson 残差 ──
y = merged["low_rating_count"].values
pearson_resid = (y - mu) / np.sqrt(mu)

# ── 6. 建立 gravity 权重矩阵 ──
msoa = gpd.read_file("0_raw/2021 London MSOA/2021_London_MSOA.shp")[["MSOA21CD", "geometry"]]
msoa = msoa[msoa["MSOA21CD"].isin(merged["MSOA21CD"])].copy()
msoa = msoa.set_index("MSOA21CD").loc[merged["MSOA21CD"].values].reset_index()

coords = np.column_stack([msoa.geometry.centroid.x.values, msoa.geometry.centroid.y.values])
dist = cdist(coords, coords)
np.fill_diagonal(dist, np.inf)
weights = 1.0 / (dist ** BETA)

W = full2W(weights, ids=msoa["MSOA21CD"].tolist())
W.transform = "r"

# ── 7. Moran's I ──
mi = Moran(pearson_resid, W)

print(f"Gravity 权重设定：w_ij = 1 / d_ij^{BETA}（行标准化）")
print("\n── GWNBR 残差 Moran's I（gravity 距离权重）──")
print(f"  Moran's I:  {mi.I:.4f}")
print(f"  期望值 E[I]: {mi.EI:.4f}")
print(f"  z-score:    {mi.z_norm:.4f}")
print(f"  p-value:    {mi.p_norm:.4f}")
print(f"  结论: {'存在显著空间自相关（GWNBR 未完全消除空间聚集）' if mi.p_norm < 0.05 else '无显著空间自相关（GWNBR 成功吸收空间结构）'}")

# ── 8. 与 NBR 对比（同一套 gravity 权重、同一套变量集）──
nbr_mi = pd.read_csv("NBR/vif_final/12b_morans_i_gravity_vif_final.csv", encoding="utf-8-sig")
print("\n── 模型对比（NBR 与 GWNBR 变量集完全一致，均为 gravity 权重）──")
print(f"  NBR   Moran's I = {nbr_mi['Moran_I'].iloc[0]:.4f}  "
      f"(z={nbr_mi['z_score'].iloc[0]:.2f}, p={nbr_mi['p_value'].iloc[0]:.4f})")
print(f"  GWNBR Moran's I = {mi.I:.4f}  (z={mi.z_norm:.2f}, p={mi.p_norm:.4f})")

# ── 9. 保存 ──
out = pd.DataFrame([{
    "Model":       "GWNBR",
    "Weight_Type": f"Gravity (w_ij = 1/d_ij^{BETA})",
    "Moran_I":     round(mi.I, 4),
    "E_I":         round(mi.EI, 4),
    "z_score":     round(mi.z_norm, 4),
    "p_value":     round(mi.p_norm, 4),
}])
out.to_csv("GWNBR/17b_gwnbr_morans_i_gravity.csv", index=False, encoding="utf-8-sig")
print("\n结果已保存：GWNBR/17b_gwnbr_morans_i_gravity.csv")
