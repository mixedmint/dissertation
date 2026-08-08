import pandas as pd
import numpy as np

df = pd.read_csv("data processing/7_regression.csv", encoding="utf-8-sig")

# ── 变量分组 ──
var_groups = {
    "IMD": {
        "IMD": "IMD Score, Pop-Weighted (Higher=More Deprived)",
    },
    "Ethnicity (%)": {
        "Asian":            "Asian (%)",
        "Black":            "Black (%)",
        "Mixed_or_Multiple":"Mixed or Multiple (%)",
        "White":            "White (%)",
        "Other":            "Other (%)",
    },
    "Population Density": {
        "Population_density": "Population Density (persons/km²)",
    },
    "Business Type (%)": {
        "restaurant_cafe_canteen":          "Restaurant / Café / Canteen (%)",
        "retailers___other":                "Other Retailers (%)",
        "takeaway_sandwich_shop":           "Takeaway & Sandwich Shops (%)",
        "pub_bar_nightclub":                "Pubs, Bars & Nightclubs (%)",
        "other_catering_premises":          "Other Catering Premises (%)",
        "supermarkets_hypermarkets":        "Supermarkets & Hypermarkets (%)",
        "hotel_bed_and_breakfast_guest_house": "Hotels & Guesthouses (%)",
    },
    "Outcome Variable": {
        "low_rating_count": "Low-Rated Outlet Count per MSOA",
        "low_rating_pct":   "Low-Rated Outlets (%, Score 0–2)",
        "restaurant_count": "Total Outlet Count per MSOA",
    },
}

# ── 计算描述性统计 ──
rows = []
for group, variables in var_groups.items():
    for col, label in variables.items():
        s = df[col].dropna()
        rows.append({
            "Group":    group,
            "Variable": label,
            "N":        int(s.count()),
            "Mean":     round(s.mean(), 2),
            "SD":       round(s.std(), 2),
            "Min":      round(s.min(), 2),
            "Q1":       round(s.quantile(0.25), 2),
            "Median":   round(s.median(), 2),
            "Q3":       round(s.quantile(0.75), 2),
            "Max":      round(s.max(), 2),
        })

stats = pd.DataFrame(rows)
stats.to_csv("data processing/descriptive_stats.csv", index=False, encoding="utf-8-sig")

# ── 打印 ──
pd.set_option("display.max_colwidth", 45)
pd.set_option("display.width", 120)
for group in stats["Group"].unique():
    print(f"\n{'─'*70}")
    print(f"  {group}")
    print(f"{'─'*70}")
    sub = stats[stats["Group"] == group].drop(columns="Group")
    print(sub.to_string(index=False))

print("\n\n结果已保存：data processing/descriptive_stats.csv")
