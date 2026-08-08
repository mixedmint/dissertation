import pandas as pd
import numpy as np
import geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran

# ── 读取数据（与 Gi* 脚本相同数据源）──
agg  = pd.read_csv("data processing/4_FHRS_agg_MSOA.csv", encoding="utf-8-sig")
msoa = gpd.read_file("0_raw/2021 London MSOA/2021_London_MSOA.shp")[["MSOA21CD", "MSOA21NM", "geometry"]]
gdf  = msoa.merge(agg[["MSOA21CD", "low_rating_pct"]], on="MSOA21CD", how="left")
gdf["low_rating_pct"] = gdf["low_rating_pct"].fillna(0)

print(f"样本量：{len(gdf):,} 个 MSOA")
print(f"low_rating_pct  均值={gdf['low_rating_pct'].mean():.3f}  "
      f"标准差={gdf['low_rating_pct'].std():.3f}\n")

# ── 空间权重矩阵（Queen Contiguity，行标准化）──
W = Queen.from_dataframe(gdf, idVariable="MSOA21CD")
W.transform = "r"
print(f"平均邻居数：{W.mean_neighbors:.2f}")

# ── Global Moran's I ──
mi = Moran(gdf["low_rating_pct"].values, W)

print("\n── Global Moran's I（low_rating_pct）──")
print(f"  Moran's I:   {mi.I:.4f}")
print(f"  期望值 E[I]: {mi.EI:.4f}")
print(f"  z-score:     {mi.z_norm:.4f}")
print(f"  p-value:     {mi.p_norm:.4f}")
print(f"  结论: {'存在显著正向空间自相关' if mi.p_norm < 0.05 and mi.I > 0 else '无显著空间自相关'}")

# ── 保存 ──
out = pd.DataFrame([{
    "Variable": "low_rating_pct",
    "Moran_I":  round(mi.I, 4),
    "E_I":      round(mi.EI, 4),
    "z_score":  round(mi.z_norm, 4),
    "p_value":  round(mi.p_norm, 4),
}])
out.to_csv("spatial analysis/getis/16a_morans_i_pct.csv", index=False, encoding="utf-8-sig")
print("\n结果已保存：spatial analysis/getis/16a_morans_i_pct.csv")
