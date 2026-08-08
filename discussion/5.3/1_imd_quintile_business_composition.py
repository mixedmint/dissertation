import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 全局模型 IMD 显著为正是否与业态构成有关：按 IMD 五分位分组看业态占比趋势 ──
# IMD 用的是人口加权 Score（越高越贫困），所以直接按数值升序切五分位即可：
# Q1 = IMD 最低（最不贫困），Q5 = IMD 最高（最贫困），和 Oldroyd 的五分位设置一致。
df = pd.read_csv("data processing/7_regression.csv", encoding="utf-8-sig")

df["IMD_quintile"] = pd.qcut(df["IMD"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

biz_cols = [
    "restaurant_cafe_canteen", "retailers___other", "takeaway_sandwich_shop",
    "pub_bar_nightclub", "other_catering_premises", "supermarkets_hypermarkets",
    "hotel_bed_and_breakfast_guest_house",
]

# ── 第一、二步：分组 + 组均值 ──
group_means = df.groupby("IMD_quintile", observed=True)[biz_cols + ["IMD"]].mean().round(2)
group_n = df.groupby("IMD_quintile", observed=True).size().rename("N")
result = pd.concat([group_n, group_means], axis=1)

print("=== 按 IMD 五分位分组，各业态占比均值 (%) ===")
print(result.to_string())
result.to_csv("discussion/5.3/1_imd_quintile_business_composition.csv", encoding="utf-8-sig")

# ── 第三步：看趋势 ──
# NBR 主模型里 5 个显著业态变量的系数符号：
#   takeaway_sandwich_shop        + （风险业态，预期随 deprivation 上升）
#   retailers___other             + （风险业态，预期随 deprivation 上升）
#   supermarkets_hypermarkets     - （保护性业态，预期随 deprivation 下降）
#   other_catering_premises       - （保护性业态，预期随 deprivation 下降）
#   hotel_bed_and_breakfast_guest_house - （保护性业态，预期随 deprivation 下降）
specs = [
    ("takeaway_sandwich_shop",               "Takeaway & Sandwich Shops", "+", True,  "risk"),
    ("retailers___other",                    "Other Retailers",           "+", True,  "risk"),
    ("supermarkets_hypermarkets",             "Supermarkets & Hypermarkets", "-", False, "protective"),
    ("other_catering_premises",               "Other Catering",            "-", False, "protective"),
    ("hotel_bed_and_breakfast_guest_house",   "Hotels & Guesthouses",      "-", False, "protective"),
]


def check_monotonic(s, increasing):
    if increasing:
        return all(s.iloc[i] <= s.iloc[i + 1] for i in range(len(s) - 1))
    return all(s.iloc[i] >= s.iloc[i + 1] for i in range(len(s) - 1))


print("\n=== 趋势判断（NBR系数符号 vs 实际是否随 deprivation 呈对应方向）===")
rows = []
for col, label, sign, expect_increasing, role in specs:
    s = group_means[col]
    mono = check_monotonic(s, expect_increasing)
    direction = "上升" if expect_increasing else "下降"
    print(f"{label}（NBR系数{sign}，{role}业态，预期随deprivation{direction}）")
    print(f"  Q1→Q5: {' -> '.join(f'{v:.2f}' for v in s)}")
    print(f"  是否单调{direction}: {'是' if mono else '否'}")
    rows.append({
        "Variable": label, "NBR_Coef_Sign": sign, "Role": role,
        "Expected_Direction": direction, "Monotonic": mono,
        "Q1": s.iloc[0], "Q5": s.iloc[4], "Q5_minus_Q1": round(s.iloc[4] - s.iloc[0], 2),
    })

trend = pd.DataFrame(rows)
trend.to_csv("discussion/5.3/1_imd_quintile_trend_summary.csv", index=False, encoding="utf-8-sig")

# ── 趋势图：5 条线放一张图里 ──
fig, ax = plt.subplots(figsize=(9, 6.5))
x = range(1, 6)
xticklabels = ["Q1\n(Least Deprived)", "Q2", "Q3", "Q4", "Q5\n(Most Deprived)"]
colors = ["#d7191c", "#e8711a", "#2c7bb6", "#2ca25f", "#756bb1"]

for (col, label, *_), color in zip(specs, colors):
    ax.plot(x, group_means[col].values, marker="o", linewidth=2, color=color, label=label)

ax.set_xticks(x); ax.set_xticklabels(xticklabels)
ax.set_xlabel("IMD Quintile", fontsize=11)
ax.set_ylabel("Mean Business Type Share (%)", fontsize=11)
ax.set_title("Business Type Composition Across IMD Quintiles", fontsize=14, fontweight="bold", pad=12)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("discussion/5.3/1_imd_quintile_trend.png", dpi=200, bbox_inches="tight")
plt.close()

print("\n结果已保存：")
print("  discussion/5.3/1_imd_quintile_business_composition.csv（各分位业态均值全表）")
print("  discussion/5.3/1_imd_quintile_trend_summary.csv（5个显著业态趋势摘要）")
print("  discussion/5.3/1_imd_quintile_trend.png（趋势图）")
