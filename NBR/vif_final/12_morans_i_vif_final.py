import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.formula.api as smf
from libpysal.weights import Queen
from esda.moran import Moran

# ── 重跑 VIF 筛选变量集的 NBR，获取残差 ──
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

# Pearson 残差
mu = result.predict()
pearson_resid = (df_clean["low_rating_count"] - mu) / np.sqrt(mu)

# ── 建立空间权重矩阵（Queen 邻接）──
msoa = gpd.read_file("0_raw/2021 London MSOA/2021_London_MSOA.shp")[["MSOA21CD", "geometry"]]
msoa = msoa[msoa["MSOA21CD"].isin(df_clean["MSOA21CD"])].copy()
msoa = msoa.set_index("MSOA21CD").loc[df_clean["MSOA21CD"].values].reset_index()

W = Queen.from_dataframe(msoa, idVariable="MSOA21CD")
W.transform = "r"  # 行标准化

# ── Moran's I ──
mi = Moran(pearson_resid.values, W)

print("── NBR（VIF筛选变量集）残差 Moran's I ──")
print(f"  Moran's I:  {mi.I:.4f}")
print(f"  期望值 E[I]: {mi.EI:.4f}")
print(f"  z-score:    {mi.z_norm:.4f}")
print(f"  p-value:    {mi.p_norm:.4f}")
print(f"  结论: {'存在显著空间自相关' if mi.p_norm < 0.05 else '无显著空间自相关'}")

out = pd.DataFrame([{
    "Moran_I":  round(mi.I, 4),
    "E_I":      round(mi.EI, 4),
    "z_score":  round(mi.z_norm, 4),
    "p_value":  round(mi.p_norm, 4),
}])
out.to_csv("NBR/vif_final/12_morans_i_vif_final.csv", index=False, encoding="utf-8-sig")
print("\n结果已保存：NBR/vif_final/12_morans_i_vif_final.csv")
