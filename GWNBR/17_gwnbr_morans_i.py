import pandas as pd
import numpy as np
import geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran

# ── 1. 读取局部系数 ──
local = pd.read_csv("GWNBR/11_gwnbr_local_coefs.csv", encoding="utf-8-sig")

# ── 2. 读取原始数据（与 R 脚本相同的预处理）──
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

# ── 5. Pearson 残差（与 NBR 脚本一致）──
y = merged["low_rating_count"].values
pearson_resid = (y - mu) / np.sqrt(mu)

print(f"\nPearson 残差统计：")
print(f"  均值:  {pearson_resid.mean():.4f}")
print(f"  标准差: {pearson_resid.std():.4f}")
print(f"  最小:  {pearson_resid.min():.4f}")
print(f"  最大:  {pearson_resid.max():.4f}")

# ── 6. 建立 Queen 邻接权重矩阵 ──
msoa = gpd.read_file("0_raw/2021 London MSOA/2021_London_MSOA.shp")[["MSOA21CD", "geometry"]]
msoa = msoa[msoa["MSOA21CD"].isin(merged["MSOA21CD"])].copy()
msoa = msoa.set_index("MSOA21CD").loc[merged["MSOA21CD"].values].reset_index()

W = Queen.from_dataframe(msoa, idVariable="MSOA21CD")
W.transform = "r"   # 行标准化

# ── 7. Moran's I ──
mi = Moran(pearson_resid, W)

print("\n── GWNBR 残差 Moran's I ──")
print(f"  Moran's I:  {mi.I:.4f}")
print(f"  期望值 E[I]: {mi.EI:.4f}")
print(f"  z-score:    {mi.z_norm:.4f}")
print(f"  p-value:    {mi.p_norm:.4f}")
print(f"  结论: {'存在显著空间自相关（GWNBR 未完全消除空间聚集）' if mi.p_norm < 0.05 else '无显著空间自相关（GWNBR 成功吸收空间结构）'}")

# ── 8. 与 NBR 对比 ──
# 注意：这里对比的是主模型（含 Black）的 NBR Moran's I，仅作整体参照；
# 并非严格的同变量集对比（主模型多了 Black 这一个控制变量）。
nbr_mi = pd.read_csv("NBR/no_age/12_morans_i.csv", encoding="utf-8-sig")
print("\n── 模型对比（NBR 为含 Black 的主模型，仅供参照）──")
print(f"  NBR   Moran's I = {nbr_mi['Moran_I'].iloc[0]:.4f}  "
      f"(z={nbr_mi['z_score'].iloc[0]:.2f}, p={nbr_mi['p_value'].iloc[0]:.4f})")
print(f"  GWNBR Moran's I = {mi.I:.4f}  (z={mi.z_norm:.2f}, p={mi.p_norm:.4f})")

# ── 9. 保存 ──
out = pd.DataFrame([{
    "Model":   "GWNBR",
    "Moran_I": round(mi.I, 4),
    "E_I":     round(mi.EI, 4),
    "z_score": round(mi.z_norm, 4),
    "p_value": round(mi.p_norm, 4),
}])
out.to_csv("GWNBR/17_gwnbr_morans_i.csv", index=False, encoding="utf-8-sig")
print("\n结果已保存：GWNBR/17_gwnbr_morans_i.csv")
