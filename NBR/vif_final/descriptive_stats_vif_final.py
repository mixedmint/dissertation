import pandas as pd

# ── 描述性统计：correlation matrix + VIF 逐步剔除后最终保留的变量集 ──
# 见 NBR/vif_stepwise_after_corr_filter_final.csv
df = pd.read_csv("data processing/7_regression.csv", encoding="utf-8-sig")

var_groups = {
    "IMD": {
        "IMD": "IMD Score, Pop-Weighted (Higher=More Deprived)",
    },
    "Ethnicity (%)": {
        "Asian": "Asian (%)",
        "Other": "Other (%)",
    },
    "Population Density": {
        "Population_density": "Population Density (persons/km²)",
    },
    "Business Type (%)": {
        "retailers___other":                   "Other Retailers (%)",
        "takeaway_sandwich_shop":               "Takeaway & Sandwich Shops (%)",
        "pub_bar_nightclub":                    "Pubs, Bars & Nightclubs (%)",
        "other_catering_premises":              "Other Catering Premises (%)",
        "supermarkets_hypermarkets":            "Supermarkets & Hypermarkets (%)",
        "hotel_bed_and_breakfast_guest_house":  "Hotels & Guesthouses (%)",
    },
}

# 描述统计报原始单位（更易解读），但 NBR 模型里实际用的是 log(Population_density)
# （缓解右偏、稳定方差），在此注明以免和回归结果表对不上
NOTES = {
    "Population_density": "Log-transformed prior to inclusion in NBR models to address right-skewness",
}

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
            "Note":     NOTES.get(col, ""),
        })

stats = pd.DataFrame(rows)
stats.to_csv("NBR/vif_final/descriptive_stats_vif_final.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.max_colwidth", 45)
pd.set_option("display.width", 120)
for group in stats["Group"].unique():
    print(f"\n{'─'*70}")
    print(f"  {group}")
    print(f"{'─'*70}")
    sub = stats[stats["Group"] == group].drop(columns=["Group", "Note"])
    print(sub.to_string(index=False))

notes = stats[stats["Note"] != ""][["Variable", "Note"]]
if len(notes):
    print(f"\n{'─'*70}")
    print("  Notes")
    print(f"{'─'*70}")
    for _, row in notes.iterrows():
        print(f"  {row['Variable']}: {row['Note']}")

print("\n\n结果已保存：NBR/vif_final/descriptive_stats_vif_final.csv")
