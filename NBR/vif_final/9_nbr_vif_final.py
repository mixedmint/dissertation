import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# ── 用 vif_stepwise_after_corr_filter.py 最终筛出的变量集跑 NBR ──
# 变量集：IMD + Asian, Other, retailers___other, takeaway_sandwich_shop,
#         pub_bar_nightclub, supermarkets_hypermarkets, other_catering_premises,
#         hotel_bed_and_breakfast_guest_house, log_pop_density
# （无 age，ethnicity 只留 Asian/Other，即去掉 Black/White/Mixed_or_Multiple）
df = pd.read_csv("data processing/7_regression.csv", encoding="utf-8-sig")

df["log_restaurant_count"] = np.log(df["restaurant_count"])
df["log_pop_density"] = np.log(df["Population_density"])

controls = [
    "Asian", "Other",
    "retailers___other", "takeaway_sandwich_shop",
    "pub_bar_nightclub", "supermarkets_hypermarkets",
    "hotel_bed_and_breakfast_guest_house", "other_catering_premises",
    "log_pop_density",
]

formula = "low_rating_count ~ IMD + " + " + ".join(controls)

cols_needed = ["low_rating_count", "IMD", "log_restaurant_count"] + controls
df_clean = df[cols_needed].dropna()
print(f"样本量：{len(df_clean)}\n")

model = smf.negativebinomial(
    formula=formula,
    data=df_clean,
    offset=df_clean["log_restaurant_count"],
)
result = model.fit(disp=False)

print(result.summary())

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
summary_df.to_csv("NBR/vif_final/9_nbr_results_vif_final.csv", index=False, encoding="utf-8-sig")

n = len(df_clean)
k = len(result.params)
aic = result.aic
aicc = aic + (2 * k * (k + 1)) / (n - k - 1)

null_model = smf.negativebinomial(
    "low_rating_count ~ 1",
    data=df_clean,
    offset=df_clean["log_restaurant_count"],
).fit(disp=False)
mcfadden_r2 = 1 - result.llf / null_model.llf

metrics = pd.DataFrame([{
    "Log_Likelihood": round(result.llf, 4),
    "AIC": round(aic, 4),
    "AICc": round(aicc, 4),
    "BIC": round(result.bic, 4),
    "McFadden_R2": round(mcfadden_r2, 4),
    "N": n,
    "K": k,
}])
metrics.to_csv("NBR/vif_final/9_nbr_metrics_vif_final.csv", index=False, encoding="utf-8-sig")

print("\n结果已保存：")
print("  NBR/vif_final/9_nbr_results_vif_final.csv")
print("  NBR/vif_final/9_nbr_metrics_vif_final.csv")
print("\n── 模型评价指标 ──")
print(metrics.to_string(index=False))
