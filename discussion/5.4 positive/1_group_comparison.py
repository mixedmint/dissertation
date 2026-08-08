import pandas as pd

# ── 为什么 GWNBR 里中南、西部 IMD 局部系数显著为正：业态构成机制的分组验证 ──
# A 组：局部 IMD 系数显著为正的 MSOA（中南、西部那批聚集区，约314个）
# B 组：其余 MSOA（不显著 + 北部显著为负的6个）
coefs = pd.read_csv("GWNBR_noblack/11_gwnbr_local_coefs.csv")[["MSOA21CD", "MSOA21NM", "IMD"]]
se = pd.read_csv("GWNBR_noblack/11_gwnbr_local_se.csv")[["MSOA21CD", "IMD"]].rename(columns={"IMD": "IMD_se"})
m = coefs.merge(se, on="MSOA21CD")
m["t"] = m["IMD"] / m["IMD_se"]
m["group"] = "B"
m.loc[(m["t"].abs() > 1.96) & (m["IMD"] > 0), "group"] = "A"

df = pd.read_csv("data processing/7_regression.csv", encoding="utf-8-sig")
df = df.merge(m[["MSOA21CD", "group"]], on="MSOA21CD", how="left")

n_a, n_b = (df["group"] == "A").sum(), (df["group"] == "B").sum()
print(f"A 组（局部IMD系数显著为正）: n = {n_a}")
print(f"B 组（其余：不显著 + 显著为负）: n = {n_b}")
print()

# ══════════════════════════════════════════════════════
# 层次一：两组业态构成对比（背景层，较弱证据）
# ══════════════════════════════════════════════════════
biz_check = ["takeaway_sandwich_shop", "retailers___other", "supermarkets_hypermarkets", "restaurant_cafe_canteen"]
tier1 = df.groupby("group")[biz_check + ["IMD"]].mean().round(2).T
tier1["A_minus_B"] = (tier1["A"] - tier1["B"]).round(2)

print("=" * 60)
print("层次一：A组 vs B组 业态构成均值对比")
print("=" * 60)
print(tier1.to_string())
tier1.to_csv("discussion/5.4 positive/1_tier1_business_composition.csv", encoding="utf-8-sig")

# ══════════════════════════════════════════════════════
# 层次二：组内 IMD 与低评分率的相关性对比（更强证据）
# ══════════════════════════════════════════════════════
r_a_rating = df.loc[df["group"] == "A", ["IMD", "low_rating_pct"]].corr().iloc[0, 1]
r_b_rating = df.loc[df["group"] == "B", ["IMD", "low_rating_pct"]].corr().iloc[0, 1]

print()
print("=" * 60)
print("层次二：组内 IMD vs low_rating_pct 相关系数")
print("=" * 60)
print(f"A 组内部: r = {r_a_rating:.4f}  (n={n_a})")
print(f"B 组内部: r = {r_b_rating:.4f}  (n={n_b})")

tier2 = pd.DataFrame([
    {"Group": "A (significant positive)", "N": n_a, "Pearson_r_IMD_vs_lowrating": round(r_a_rating, 4)},
    {"Group": "B (rest)", "N": n_b, "Pearson_r_IMD_vs_lowrating": round(r_b_rating, 4)},
])
tier2.to_csv("discussion/5.4 positive/2_tier2_imd_lowrating_correlation.csv", index=False, encoding="utf-8-sig")

# ══════════════════════════════════════════════════════
# 层次三：组内 IMD 与高风险业态（takeaway / other retailers）的相关性对比
# ══════════════════════════════════════════════════════
rows = []
for col, label in [("takeaway_sandwich_shop", "Takeaway & Sandwich Shops"),
                    ("retailers___other", "Other Retailers")]:
    r_a = df.loc[df["group"] == "A", ["IMD", col]].corr().iloc[0, 1]
    r_b = df.loc[df["group"] == "B", ["IMD", col]].corr().iloc[0, 1]
    rows.append({"Business_Type": label, "r_A_group": round(r_a, 4), "r_B_group": round(r_b, 4),
                 "A_minus_B": round(r_a - r_b, 4)})

tier3 = pd.DataFrame(rows)
print()
print("=" * 60)
print("层次三：组内 IMD vs 高风险业态占比 相关系数")
print("=" * 60)
print(tier3.to_string(index=False))
tier3.to_csv("discussion/5.4 positive/3_tier3_imd_risktype_correlation.csv", index=False, encoding="utf-8-sig")

print("\n结果已保存：")
print("  discussion/5.4 positive/1_tier1_business_composition.csv")
print("  discussion/5.4 positive/2_tier2_imd_lowrating_correlation.csv")
print("  discussion/5.4 positive/3_tier3_imd_risktype_correlation.csv")
