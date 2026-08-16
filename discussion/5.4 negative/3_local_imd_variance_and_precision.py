import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree

# ── 负显著 MSOA 的"局部 IMD 方差"（186近邻窗口内 IMD 的离散程度）
#    以及局部系数标准误（SE），分别和全城1002个MSOA的分布对比 ──
neg_list = pd.read_csv("discussion/5.4 negative/1_negative_msoa_list.csv")
targets = neg_list["MSOA21NM"].tolist()

df = pd.read_csv("data processing/7_regression.csv", encoding="utf-8-sig")
msoa = gpd.read_file("0_raw/2021 London MSOA/2021_London_MSOA.shp")[["MSOA21CD", "geometry"]]
msoa["x"] = msoa.geometry.centroid.x
msoa["y"] = msoa.geometry.centroid.y
df = df.merge(msoa[["MSOA21CD", "x", "y"]], on="MSOA21CD", how="left").dropna(subset=["x", "y", "IMD"]).reset_index(drop=True)

coords = df[["x", "y"]].values
tree = cKDTree(coords)
K = 186  # 主模型带宽（近邻数）

local_var = []
for i in range(len(df)):
    _, idx = tree.query(coords[i], k=K)
    local_var.append(df["IMD"].values[idx].var())
df["local_imd_var"] = local_var

sub = df[df["MSOA21NM"].isin(targets)][["MSOA21NM", "IMD", "local_imd_var"]].set_index("MSOA21NM").loc[targets]
sub["percentile_in_city"] = [round((df["local_imd_var"] < v).mean() * 100, 1) for v in sub["local_imd_var"]]

print(f"=== 局部 IMD 方差（{K}近邻窗口）：负显著 MSOA vs 全城百分位 ===")
print(sub.round(4).to_string())
print()
print("全城 local_imd_var 描述统计:")
print(df["local_imd_var"].describe().round(4))
sub.round(4).to_csv("discussion/5.4 negative/3_local_imd_variance.csv", encoding="utf-8-sig")

# ── 局部系数标准误（精度）──
coefs = pd.read_csv("GWNBR/11_gwnbr_local_coefs.csv")[["MSOA21CD", "MSOA21NM"]]
se = pd.read_csv("GWNBR/11_gwnbr_local_se.csv").merge(coefs, on="MSOA21CD")
se_sub = se[se["MSOA21NM"].isin(targets)][["MSOA21NM", "IMD"]].set_index("MSOA21NM").loc[targets]
se_sub.columns = ["IMD_local_SE"]
se_sub["IMD_local_variance_SE2"] = se_sub["IMD_local_SE"] ** 2
se_sub["percentile_in_city"] = [round((se["IMD"] < v).mean() * 100, 1) for v in se_sub["IMD_local_SE"]]

print()
print("=== 局部系数标准误（精度）：负显著 MSOA vs 全城百分位 ===")
print(se_sub.round(6).to_string())
print()
print("全城 local SE 描述统计:")
print(se["IMD"].describe().round(4))
se_sub.round(6).to_csv("discussion/5.4 negative/3_local_coef_se.csv", encoding="utf-8-sig")

print("\n结果已保存：discussion/5.4 negative/3_local_imd_variance.csv、3_local_coef_se.csv")
