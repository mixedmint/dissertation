import pandas as pd

BUSINESS_TYPES = [
    "Restaurant/Cafe/Canteen",
    "Retailers - other",
    "Takeaway/sandwich shop",
    "Pub/bar/nightclub",
    "Other catering premises",
    "Retailers - supermarkets/hypermarkets",
    "Hotel/bed & breakfast/guest house",
]

def aggregate(df, code_col, name_col, output_file):
    df["is_low"] = (df["RatingValue"] <= 2).astype(int)

    result = (
        df.groupby([code_col, name_col])
        .agg(
            restaurant_count=("RatingValue", "count"),
            low_rating_count=("is_low", "sum"),
        )
        .reset_index()
    )
    result["low_rating_pct"] = (
        result["low_rating_count"] / result["restaurant_count"] * 100
    ).round(2)

    # 各 BusinessType 占比
    for btype in BUSINESS_TYPES:
        col = btype.lower().replace("/", "_").replace(" ", "_").replace("-", "_").replace("&", "and")
        df[col] = (df["BusinessType"] == btype).astype(int)
        counts = df.groupby(code_col)[col].sum().reset_index(name=f"{col}_count")
        result = result.merge(counts, on=code_col, how="left")
        result[f"{col}_pct"] = (result[f"{col}_count"] / result["restaurant_count"] * 100).round(2)
        result = result.drop(columns=[f"{col}_count"])

    result.to_csv(output_file, index=False, encoding="utf-8-sig")
    return result

# ── LSOA ──
lsoa_df = pd.read_csv("3_FHRS_LSOA.csv", encoding="utf-8-sig")
lsoa_result = aggregate(lsoa_df, "LSOA21CD", "LSOA21NM", "4_FHRS_agg_LSOA.csv")

# ── MSOA ──
msoa_df = pd.read_csv("3_FHRS_MSOA.csv", encoding="utf-8-sig")
msoa_result = aggregate(msoa_df, "MSOA21CD", "MSOA21NM", "4_FHRS_agg_MSOA.csv")

print("完成！")
print(f"  4_FHRS_agg_LSOA.csv:  {len(lsoa_result):,} 个 LSOA")
print(f"  4_FHRS_agg_MSOA.csv:  {len(msoa_result):,} 个 MSOA")
print("\nLSOA 前5行示例:")
print(lsoa_result.head().to_string(index=False))
print("\nMSOA 前5行示例:")
print(msoa_result.head().to_string(index=False))
