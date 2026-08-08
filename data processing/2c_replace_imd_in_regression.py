import os
import shutil
import pandas as pd

REGRESSION_FILE = "data processing/7_regression.csv"
BACKUP_FILE      = "data processing/7_regression_backup_decile_imd.csv"
IMD_SCORE_FILE   = "data processing/2b_IMD_Score_MSOA_London.csv"

# 备份原始文件（含旧的 decile-mean IMD），只在第一次运行时备份
if not os.path.exists(BACKUP_FILE):
    shutil.copy(REGRESSION_FILE, BACKUP_FILE)
    print(f"已备份原始文件：{BACKUP_FILE}")
else:
    print(f"备份文件已存在，跳过备份：{BACKUP_FILE}")

df       = pd.read_csv(REGRESSION_FILE, encoding="utf-8-sig")
col_order = df.columns.tolist()

imd_score = pd.read_csv(IMD_SCORE_FILE, encoding="utf-8-sig")[["MSOA21CD", "IMD_Score_PopWeighted"]]

before_n = df["IMD"].notna().sum()

df = df.drop(columns=["IMD"]).merge(imd_score, on="MSOA21CD", how="left")
df = df.rename(columns={"IMD_Score_PopWeighted": "IMD"})
df = df[col_order]  # 保持原列顺序不变

after_n = df["IMD"].notna().sum()
no_match = df["IMD"].isna().sum()

df.to_csv(REGRESSION_FILE, index=False, encoding="utf-8-sig")

print("完成！IMD 列已替换为人口加权 IMD Score")
print(f"  替换前非缺失行数: {before_n:,}")
print(f"  替换后非缺失行数: {after_n:,}")
if no_match:
    print(f"  警告：{no_match} 个 MSOA 未匹配到新 IMD Score")
print(f"  输出文件: {REGRESSION_FILE}")
print(df[["MSOA21CD", "IMD"]].head())
