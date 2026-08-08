import pandas as pd

# ── 带宽敏感性检验：186（主结果）与 140、240 的局部 IMD 系数做 Pearson 相关 ──
c140 = pd.read_csv("GWNBR_noblack/bandwidth=140/11_gwnbr_local_coefs.csv")[["MSOA21CD", "IMD"]].rename(columns={"IMD": "IMD_140"})
c186 = pd.read_csv("GWNBR_noblack/11_gwnbr_local_coefs.csv")[["MSOA21CD", "IMD"]].rename(columns={"IMD": "IMD_186"})
c240 = pd.read_csv("GWNBR_noblack/bandwidth=240/11_gwnbr_local_coefs.csv")[["MSOA21CD", "IMD"]].rename(columns={"IMD": "IMD_240"})

m = c186.merge(c140, on="MSOA21CD").merge(c240, on="MSOA21CD")
print(f"匹配样本量: {len(m)}")

r_140 = m["IMD_186"].corr(m["IMD_140"])
r_240 = m["IMD_186"].corr(m["IMD_240"])

print(f"Pearson r (186 vs 140): {r_140:.4f}")
print(f"Pearson r (186 vs 240): {r_240:.4f}")

out = pd.DataFrame([
    {"Comparison": "186 vs 140", "N": len(m), "Pearson_r": round(r_140, 4)},
    {"Comparison": "186 vs 240", "N": len(m), "Pearson_r": round(r_240, 4)},
])
out.to_csv("GWNBR_noblack/bandwidth_sensitivity_correlation.csv", index=False, encoding="utf-8-sig")
print("\n结果已保存：GWNBR_noblack/bandwidth_sensitivity_correlation.csv")
