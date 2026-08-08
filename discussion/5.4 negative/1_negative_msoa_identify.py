import pandas as pd

# ── 识别 GWNBR 主模型（h=186, 无Black）里 IMD 局部系数显著为负的 MSOA ──
coefs = pd.read_csv("GWNBR_noblack/11_gwnbr_local_coefs.csv")
se    = pd.read_csv("GWNBR_noblack/11_gwnbr_local_se.csv")[["MSOA21CD", "IMD"]].rename(columns={"IMD": "IMD_se"})

m = coefs[["MSOA21CD", "MSOA21NM", "IMD"]].merge(se, on="MSOA21CD")
m["t"] = m["IMD"] / m["IMD_se"]
m["sig"] = m["t"].abs() > 1.96

neg = m[m["sig"] & (m["IMD"] < 0)].sort_values("IMD").copy()
pos = m[m["sig"] & (m["IMD"] > 0)].copy()

print(f"全部显著 MSOA 数: {m['sig'].sum()}")
print(f"  其中系数为负: {len(neg)}")
print(f"  其中系数为正: {len(pos)}")
print()
print("── 显著为负的 MSOA ──")
print(neg[["MSOA21NM", "IMD", "IMD_se", "t"]].round(6).to_string(index=False))

neg.to_csv("discussion/5.4 negative/1_negative_msoa_list.csv", index=False, encoding="utf-8-sig")
print("\n结果已保存：discussion/5.4 negative/1_negative_msoa_list.csv")
