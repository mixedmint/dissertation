import pandas as pd

# ── 负显著 MSOA 的餐饮店总数（排除小样本导致 low_rating_pct 本身不稳定的可能）──
neg_list = pd.read_csv("discussion/5.4 negative/1_negative_msoa_list.csv")
targets = neg_list["MSOA21NM"].tolist()

df = pd.read_csv("data processing/7_regression.csv", encoding="utf-8-sig")
sub = df[df["MSOA21NM"].isin(targets)][
    ["MSOA21NM", "restaurant_count", "low_rating_count", "low_rating_pct"]
].set_index("MSOA21NM").loc[targets]

print("=== 负显著 MSOA 的餐饮店总数与低分店占比 ===")
print(sub.to_string())
print()
print("全城 restaurant_count 描述统计:")
print(df["restaurant_count"].describe())

sub.to_csv("discussion/5.4 negative/4_sample_size_check.csv", encoding="utf-8-sig")
print("\n结果已保存：discussion/5.4 negative/4_sample_size_check.csv")
