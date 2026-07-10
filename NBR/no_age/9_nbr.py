import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_csv("data processing/7_regression.csv", encoding="utf-8-sig")

# log offset
df["log_restaurant_count"] = np.log(df["restaurant_count"])
df["log_pop_density"] = np.log(df["Population_density"])

# 列名中含特殊字符，重命名以兼容 formula
df = df.rename(columns={
    "0-4":   "age_0_4",
    "65plus": "age_65plus",
})

# 变量列表（参照组：White, Mixed_or_Multiple，restaurant_cafe_canteen；无 age 变量）
controls = [
    "Asian", "Black", "Other",
    "retailers___other", "takeaway_sandwich_shop",
    "pub_bar_nightclub", "supermarkets_hypermarkets",
    "hotel_bed_and_breakfast_guest_house", "other_catering_premises",
    "log_pop_density",
]

formula = "low_rating_count ~ IMD + " + " + ".join(controls)

# 删除缺失值
cols_needed = ["low_rating_count", "IMD", "log_restaurant_count"] + controls
df_clean = df[cols_needed].dropna()
print(f"样本量：{len(df_clean)}\n")

# ── Negative Binomial Regression ──
model = smf.negativebinomial(
    formula=formula,
    data=df_clean,
    offset=df_clean["log_restaurant_count"],
)
result = model.fit(disp=False)

print(result.summary())

# 保存结果
def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

summary_df = pd.DataFrame({
    "Variable":  result.params.index,
    "Coef":      result.params.values.round(4),
    "Std_Err":   result.bse.values.round(4),
    "z":         result.tvalues.values.round(4),
    "p_value":   result.pvalues.values.round(4),
    "Sig":       [stars(p) for p in result.pvalues],
    "CI_lower":  result.conf_int()[0].values.round(4),
    "CI_upper":  result.conf_int()[1].values.round(4),
    "IRR":       np.exp(result.params.values).round(4),
})
summary_df.to_csv("NBR/9_nbr_results_no_age.csv", index=False, encoding="utf-8-sig")

# ── 模型评价指标 ──
n = len(df_clean)
k = len(result.params)
aic  = result.aic
aicc = aic + (2 * k * (k + 1)) / (n - k - 1)

# McFadden's pseudo-R²：需要 null model（只含截距）
null_model = smf.negativebinomial(
    "low_rating_count ~ 1",
    data=df_clean,
    offset=df_clean["log_restaurant_count"],
).fit(disp=False)
mcfadden_r2 = 1 - result.llf / null_model.llf

metrics = pd.DataFrame([{
    "Log_Likelihood":  round(result.llf, 4),
    "AIC":             round(aic, 4),
    "AICc":            round(aicc, 4),
    "BIC":             round(result.bic, 4),
    "McFadden_R2":     round(mcfadden_r2, 4),
    "N":               n,
    "K":               k,
}])
metrics.to_csv("NBR/9_nbr_metrics_no_age.csv", index=False, encoding="utf-8-sig")

print("\n结果已保存：NBR/9_nbr_results_no_age.csv、NBR/9_nbr_metrics_no_age.csv")
print("\n── 模型评价指标 ──")
print(metrics.to_string(index=False))
