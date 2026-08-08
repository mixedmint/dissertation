import pandas as pd

# ── 负显著 MSOA 的业态构成 vs 全城中位数 ──
neg_list = pd.read_csv("discussion/5.4 negative/1_negative_msoa_list.csv")
targets = neg_list["MSOA21NM"].tolist()

df = pd.read_csv("data processing/7_regression.csv", encoding="utf-8-sig")

biz_cols = ["restaurant_cafe_canteen", "retailers___other", "takeaway_sandwich_shop",
            "pub_bar_nightclub", "other_catering_premises", "supermarkets_hypermarkets",
            "hotel_bed_and_breakfast_guest_house"]

sub = df[df["MSOA21NM"].isin(targets)][["MSOA21NM", "IMD"] + biz_cols].set_index("MSOA21NM").loc[targets]
city_median = df[biz_cols].median()
city_median.name = "City_Median"

diff = sub[biz_cols].sub(city_median, axis=1)

print("=== 负显著 MSOA 的业态占比 (%) ===")
print(sub[biz_cols].round(2).to_string())
print()
print("=== 全城各业态中位数 (%) ===")
print(city_median.round(2).to_string())
print()
print("=== 差值（MSOA - 全城中位数），正=高于中位数 ===")
print(diff.round(2).to_string())

sub[biz_cols].round(2).to_csv("discussion/5.4 negative/2_business_type_composition.csv", encoding="utf-8-sig")
diff.round(2).to_csv("discussion/5.4 negative/2_business_type_diff_from_median.csv", encoding="utf-8-sig")
city_median.round(2).to_csv("discussion/5.4 negative/2_business_type_city_median.csv", encoding="utf-8-sig")
print("\n结果已保存：discussion/5.4 negative/2_business_type_*.csv")
