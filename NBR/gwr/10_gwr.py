import pandas as pd
import numpy as np
import geopandas as gpd
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import statsmodels.api as sm

# ── 读取数据 ──
df = pd.read_csv("data processing/7_regression.csv", encoding="utf-8-sig")
df = df.rename(columns={"0-4": "age_0_4", "65plus": "age_65plus"})

# ── 提取 MSOA 质心坐标（EPSG:27700，单位：米）──
msoa = gpd.read_file("0_raw/2021 London MSOA/2021_London_MSOA.shp")[["MSOA21CD", "geometry"]]
msoa["x"] = msoa.geometry.centroid.x
msoa["y"] = msoa.geometry.centroid.y
msoa = msoa[["MSOA21CD", "x", "y"]]

df = df.merge(msoa, on="MSOA21CD", how="left")

# population density log 变换（与 NBR 保持一致）
df["log_pop_density"] = np.log(df["Population_density"])

# ── 变量设置（无 age 变量，参照组：White+Mixed, restaurant_cafe_canteen）──
y_col = "low_rating_pct"
x_cols = [
    "IMD",
    "Asian", "Black", "Other",
    "retailers___other", "takeaway_sandwich_shop",
    "pub_bar_nightclub", "supermarkets_hypermarkets",
    "hotel_bed_and_breakfast_guest_house", "other_catering_premises",
    "log_pop_density",
]

cols_needed = [y_col] + x_cols + ["x", "y", "MSOA21CD"]
df_clean = df[cols_needed].dropna()
print(f"样本量：{len(df_clean)}\n")

coords = list(zip(df_clean["x"], df_clean["y"]))
y = df_clean[[y_col]].values
X = df_clean[x_cols].values

# ── 带宽选择（AICc，自适应）──
print("正在选择最优带宽（adaptive, AICc）...")
selector = Sel_BW(coords, y, X, kernel="bisquare", fixed=False)
bw = selector.search()
print(f"最优带宽（邻居数）：{bw}\n")

# ── 运行 GWR ──
print("正在运行 GWR...")
model = GWR(coords, y, X, bw=bw, kernel="bisquare", fixed=False)
result = model.fit()

print(result.summary())

# ── 全局 OLS 系数（带变量名和显著性）──
X_ols = sm.add_constant(df_clean[x_cols])
ols = sm.OLS(df_clean[y_col], X_ols).fit()

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

global_df = pd.DataFrame({
    "Variable": ["Intercept"] + x_cols,
    "Coef":     ols.params.values.round(4),
    "Std_Err":  ols.bse.values.round(4),
    "t":        ols.tvalues.values.round(4),
    "p_value":  ols.pvalues.values.round(4),
    "Sig":      [stars(p) for p in ols.pvalues],
})
global_df.to_csv("NBR/10_gwr_global_coefs.csv", index=False, encoding="utf-8-sig")

# ── 模型指标 ──
metrics = pd.DataFrame([{
    "Bandwidth":  bw,
    "R2":         round(result.R2, 4),
    "Adj_R2":     round(result.adj_R2, 4),
    "AICc":       round(result.aicc, 4),
    "AIC":        round(result.aic, 4),
    "BIC":        round(result.bic, 4),
}])
metrics.to_csv("NBR/10_gwr_metrics.csv", index=False, encoding="utf-8-sig")

# ── 局部系数 ──
coef_cols = ["Intercept"] + x_cols
coef_df = pd.DataFrame(result.params, columns=coef_cols)
coef_df.insert(0, "MSOA21CD", df_clean["MSOA21CD"].values)
coef_df["local_R2"] = result.localR2
coef_df["y_pred"]   = result.predy.flatten()
coef_df.to_csv("NBR/10_gwr_local_coefs.csv", index=False, encoding="utf-8-sig")

print("\n── 模型指标 ──")
print(metrics.to_string(index=False))
print("\n结果已保存：")
print("  NBR/10_gwr_global_coefs.csv（全局系数）")
print("  NBR/10_gwr_metrics.csv（模型指标）")
print("  NBR/10_gwr_local_coefs.csv（局部系数）")
