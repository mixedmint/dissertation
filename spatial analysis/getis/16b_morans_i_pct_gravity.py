import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial.distance import cdist
from libpysal.weights.util import full2W
from esda.moran import Moran

# ── Global Moran's I（low_rating_pct），用 gravity-like 距离权重矩阵替代 16a 的 Queen 邻接权重 ──
# 权重定义：w_ij = 1 / d_ij^BETA（i≠j 时，d_ij 为 MSOA 质心间欧氏距离；自身权重为 0）
# 这是空间统计里最常见的重力型（gravity-type / distance-decay）权重设定，
# BETA=1 为简单反距离；如需更陡的衰减（更接近物理引力平方反比），可改成 2。
BETA = 1

# ── 读取数据（与 16a 相同数据源）──
agg  = pd.read_csv("data processing/4_FHRS_agg_MSOA.csv", encoding="utf-8-sig")
msoa = gpd.read_file("0_raw/2021 London MSOA/2021_London_MSOA.shp")[["MSOA21CD", "MSOA21NM", "geometry"]]
gdf  = msoa.merge(agg[["MSOA21CD", "low_rating_pct"]], on="MSOA21CD", how="left")
gdf["low_rating_pct"] = gdf["low_rating_pct"].fillna(0)

print(f"样本量：{len(gdf):,} 个 MSOA")
print(f"low_rating_pct  均值={gdf['low_rating_pct'].mean():.3f}  标准差={gdf['low_rating_pct'].std():.3f}\n")

# ── 质心坐标（EPSG:27700，单位：米）──
coords = np.column_stack([gdf.geometry.centroid.x.values, gdf.geometry.centroid.y.values])

# ── 全对全欧氏距离矩阵，构建 gravity 权重 ──
dist = cdist(coords, coords)
np.fill_diagonal(dist, np.inf)          # 自身距离设为 inf，对应自身权重为 0（不能自己做自己的邻居）
weights = 1.0 / (dist ** BETA)

W = full2W(weights, ids=gdf["MSOA21CD"].tolist())
W.transform = "r"                       # 行标准化，和项目里其他 Moran's I 脚本保持一致

print(f"Gravity 权重设定：w_ij = 1 / d_ij^{BETA}（距离单位：米，行标准化后单位不影响结果）")
print(f"平均每个 MSOA 的（非零权重）邻居数：{W.mean_neighbors:.0f}（gravity 权重下全部 MSOA 互为邻居，仅权重大小不同）\n")

# ── Global Moran's I ──
mi = Moran(gdf["low_rating_pct"].values, W)

print("── Global Moran's I（low_rating_pct，gravity-like 距离权重）──")
print(f"  Moran's I:   {mi.I:.4f}")
print(f"  期望值 E[I]: {mi.EI:.4f}")
print(f"  z-score:     {mi.z_norm:.4f}")
print(f"  p-value:     {mi.p_norm:.4f}")
print(f"  结论: {'存在显著正向空间自相关' if mi.p_norm < 0.05 and mi.I > 0 else '无显著空间自相关'}")

# ── 与 16a（Queen 邻接）结果对比 ──
try:
    queen_result = pd.read_csv("spatial analysis/getis/16a_morans_i_pct.csv", encoding="utf-8-sig")
    print("\n── 与 Queen 邻接权重（16a）对比 ──")
    print(f"  Queen   Moran's I = {queen_result['Moran_I'].iloc[0]:.4f}  "
          f"(z={queen_result['z_score'].iloc[0]:.2f}, p={queen_result['p_value'].iloc[0]:.4f})")
    print(f"  Gravity Moran's I = {mi.I:.4f}  (z={mi.z_norm:.2f}, p={mi.p_norm:.4f})")
except FileNotFoundError:
    pass

# ── 保存 ──
out = pd.DataFrame([{
    "Variable":       "low_rating_pct",
    "Weight_Type":    f"Gravity (w_ij = 1/d_ij^{BETA})",
    "Moran_I":        round(mi.I, 4),
    "E_I":            round(mi.EI, 4),
    "z_score":        round(mi.z_norm, 4),
    "p_value":        round(mi.p_norm, 4),
}])
out.to_csv("spatial analysis/getis/16b_morans_i_pct_gravity.csv", index=False, encoding="utf-8-sig")
print("\n结果已保存：spatial analysis/getis/16b_morans_i_pct_gravity.csv")
