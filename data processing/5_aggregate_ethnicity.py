import pandas as pd

df = pd.read_csv("0_raw/2021 ethnic group.csv")
df.columns = ["MSOA21CD", "MSOA21NM", "ethnic_code", "ethnic_group", "count"]

# 排除"Does not apply"
df = df[df["ethnic_group"] != "Does not apply"]

# 映射小类到大类
def map_group(name):
    if name.startswith("Asian"):
        return "Asian"
    elif name.startswith("Black"):
        return "Black"
    elif name.startswith("Mixed"):
        return "Mixed_or_Multiple"
    elif name.startswith("White"):
        return "White"
    else:
        return "Other"

df["broad_group"] = df["ethnic_group"].apply(map_group)

# 按 MSOA + 大类汇总
grouped = (
    df.groupby(["MSOA21CD", "MSOA21NM", "broad_group"])["count"]
    .sum()
    .reset_index()
)

# 透视为宽格式：每个 MSOA 一行
wide = grouped.pivot_table(
    index=["MSOA21CD", "MSOA21NM"],
    columns="broad_group",
    values="count",
    fill_value=0
).reset_index()
wide.columns.name = None

# 计算总人口和各族裔占比
group_cols = ["Asian", "Black", "Mixed_or_Multiple", "White", "Other"]
wide["total_population"] = wide[group_cols].sum(axis=1)

for col in group_cols:
    wide[f"{col}_pct"] = (wide[col] / wide["total_population"] * 100).round(2)

wide.to_csv("5_ethnicity_MSOA.csv", index=False, encoding="utf-8-sig")

print("完成！")
print(f"  MSOA 数量: {len(wide):,}")
print(wide.head().to_string(index=False))
