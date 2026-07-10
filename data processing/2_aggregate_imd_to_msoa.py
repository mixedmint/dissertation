import pandas as pd

IMD_FILE    = "0_raw/File_1_IoD2025 Index of Multiple Deprivation.csv"
LOOKUP_FILE = "0_raw/Output_Area_to_Lower_layer_Super_Output_Area_to_Middle_layer_Super_Output_Area_to_Local_Authority_District_(December_2021)_Lookup_in_England_and_Wales_v3.csv"
OUTPUT_FILE = "2_IMD_MSOA_London.csv"

LONDON_LAD_CODES = {
    "E09000001","E09000002","E09000003","E09000004","E09000005",
    "E09000006","E09000007","E09000008","E09000009","E09000010",
    "E09000011","E09000012","E09000013","E09000014","E09000015",
    "E09000016","E09000017","E09000018","E09000019","E09000020",
    "E09000021","E09000022","E09000023","E09000024","E09000025",
    "E09000026","E09000027","E09000028","E09000029","E09000030",
    "E09000031","E09000032","E09000033",
}

# 读取 IMD，只保留 LSOA 代码和 Decile
imd = pd.read_csv(IMD_FILE, usecols=[
    "LSOA code (2021)",
    "Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)"
])
imd.columns = ["LSOA21CD", "IMD_Decile"]

# 读取 lookup，去重得到唯一 LSOA→MSOA 映射
lookup = pd.read_csv(LOOKUP_FILE, usecols=["LSOA21CD", "MSOA21CD", "MSOA21NM", "LAD22CD"])
lookup = lookup.drop_duplicates(subset=["LSOA21CD"])

# 合并 IMD 与 lookup
merged = imd.merge(lookup, on="LSOA21CD", how="left")

# 筛选伦敦
london = merged[merged["LAD22CD"].isin(LONDON_LAD_CODES)].copy()

# 按 MSOA 聚合：计算 Decile 均值和 LSOA 数量
msoa_imd = (
    london.groupby(["MSOA21CD", "MSOA21NM"])
    .agg(
        IMD_Decile_Mean=("IMD_Decile", "mean"),
        LSOA_Count=("LSOA21CD", "count"),
    )
    .reset_index()
)

msoa_imd.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("完成！")
print(f"  伦敦 LSOA 数:  {len(london):,}")
print(f"  伦敦 MSOA 数:  {len(msoa_imd):,}")
print(f"  输出文件:      {OUTPUT_FILE}")
print(msoa_imd.head())
