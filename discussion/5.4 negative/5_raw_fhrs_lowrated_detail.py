import pandas as pd

# ── 对负显著 MSOA 里样本量不算小的那几个（餐饮店数 >= 全城中位数），
#    回到原始 FHRS 数据，看具体是哪些店把它们的评级拉低的 ──
neg_list = pd.read_csv("discussion/5.4 negative/1_negative_msoa_list.csv")
sample = pd.read_csv("discussion/5.4 negative/4_sample_size_check.csv")

city_median_n = pd.read_csv("data processing/7_regression.csv", encoding="utf-8-sig")["restaurant_count"].median()
big_enough = sample[sample["restaurant_count"] >= city_median_n]["MSOA21NM"].tolist()
print(f"全城餐饮店数中位数: {city_median_n}")
print(f"样本量不算小、纳入细看的 MSOA: {big_enough}")
print(f"（被排除的小样本 MSOA: {[m for m in neg_list['MSOA21NM'] if m not in big_enough]}，"
      f"占比本身对个别店铺评级过于敏感，不适合做业态层面解读）\n")

msoa_data = pd.read_csv("data processing/3_FHRS_MSOA.csv", encoding="utf-8-sig")
low = msoa_data[(msoa_data["MSOA21NM"].isin(big_enough)) & (msoa_data["RatingValue"] <= 2)]

raw = pd.read_csv("0_raw/FHRS_All_en-GB.csv", encoding="utf-8-sig", low_memory=False)
detail = raw[raw["FHRSID"].isin(low["FHRSID"])][
    ["FHRSID", "BusinessName", "BusinessType", "RatingValue", "RatingDate", "PostCode"]
].merge(low[["FHRSID", "MSOA21NM"]], on="FHRSID").sort_values(["MSOA21NM", "RatingValue"])

print("=== 低分店明细 ===")
print(detail.to_string(index=False))
detail.to_csv("discussion/5.4 negative/5_raw_fhrs_lowrated_detail.csv", index=False, encoding="utf-8-sig")

# ── 业态构成对比：这批低分店 vs 全城所有低分店的业态基线 ──
local_biz_pct = (detail["BusinessType"].value_counts(normalize=True) * 100).round(1)

low_all = msoa_data[msoa_data["RatingValue"] <= 2]
city_biz_pct = (low_all["BusinessType"].value_counts(normalize=True) * 100).round(1)

compare = pd.DataFrame({
    "This_Cluster_Pct": local_biz_pct,
    "Citywide_LowRated_Pct": city_biz_pct,
}).fillna(0)
compare["Diff"] = (compare["This_Cluster_Pct"] - compare["Citywide_LowRated_Pct"]).round(1)

print(f"\n=== 业态构成对比：本簇低分店（n={len(detail)}） vs 全城低分店基线（n={len(low_all)}） ===")
print(compare.to_string())
compare.to_csv("discussion/5.4 negative/5_business_type_vs_citywide_lowrated.csv", encoding="utf-8-sig")

print("\n结果已保存：")
print("  discussion/5.4 negative/5_raw_fhrs_lowrated_detail.csv")
print("  discussion/5.4 negative/5_business_type_vs_citywide_lowrated.csv")
