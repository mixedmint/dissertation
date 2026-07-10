import pandas as pd
import re

df = pd.read_csv("0_raw/2021 age.csv")
df.columns = ["MSOA21CD", "MSOA21NM", "age_code", "age_label", "count"]

# 从标签中提取数字年龄
def extract_age(label):
    if "under 1" in label:
        return 0
    if "100 years and over" in label:
        return 100
    m = re.search(r"(\d+)", label)
    return int(m.group(1)) if m else None

df["age"] = df["age_label"].apply(extract_age)

# 映射到年龄段
def map_band(age):
    if age is None:
        return None
    if age <= 4:
        return "age_0_4"
    elif age <= 14:
        return "age_5_14"
    elif age <= 19:
        return "age_15_19"
    elif age <= 24:
        return "age_20_24"
    elif age <= 44:
        return "age_25_44"
    elif age <= 64:
        return "age_45_64"
    else:
        return "age_65plus"

df["age_band"] = df["age"].apply(map_band)
df = df.dropna(subset=["age_band"])

# 按 MSOA + 年龄段汇总
grouped = (
    df.groupby(["MSOA21CD", "MSOA21NM", "age_band"])["count"]
    .sum()
    .reset_index()
)

# 透视为宽格式
band_order = ["age_0_4", "age_5_14", "age_15_19", "age_20_24",
              "age_25_44", "age_45_64", "age_65plus"]

wide = grouped.pivot_table(
    index=["MSOA21CD", "MSOA21NM"],
    columns="age_band",
    values="count",
    fill_value=0
).reset_index()
wide.columns.name = None
wide = wide[["MSOA21CD", "MSOA21NM"] + band_order]

# 总人口和占比
wide["total_population"] = wide[band_order].sum(axis=1)

for col in band_order:
    wide[f"{col}_pct"] = (wide[col] / wide["total_population"] * 100).round(2)

wide.to_csv("6_age_MSOA.csv", index=False, encoding="utf-8-sig")

print("完成！")
print(f"  MSOA 数量: {len(wide):,}")
print(wide.head().to_string(index=False))
