import pandas as pd

df = pd.read_csv("data processing/3_FHRS_MSOA.csv", encoding="utf-8-sig")

low  = df[df["RatingValue"] <= 2].copy()
high = df[df["RatingValue"] >= 3].copy()

low.to_csv("spatial analysis/13_low_rated.csv",  index=False, encoding="utf-8-sig")
high.to_csv("spatial analysis/13_high_rated.csv", index=False, encoding="utf-8-sig")

print("完成！")
print(f"  低分 (0-2): {len(low):,}  →  13_low_rated.csv")
print(f"  高分 (3-5): {len(high):,}  →  13_high_rated.csv")
