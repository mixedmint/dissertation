import pandas as pd

IMD_SCORE_FILE = "0_raw/File_5_IoD2025_Scores_for_the_Indices_of_Deprivation.csv"
LOOKUP_FILE    = "0_raw/Output_Area_to_Lower_layer_Super_Output_Area_to_Middle_layer_Super_Output_Area_to_Local_Authority_District_(December_2021)_Lookup_in_England_and_Wales_v3.csv"
OUTPUT_FILE    = "data processing/2b_IMD_Score_MSOA_London.csv"

LONDON_LAD_CODES = {
    "E09000001","E09000002","E09000003","E09000004","E09000005",
    "E09000006","E09000007","E09000008","E09000009","E09000010",
    "E09000011","E09000012","E09000013","E09000014","E09000015",
    "E09000016","E09000017","E09000018","E09000019","E09000020",
    "E09000021","E09000022","E09000023","E09000024","E09000025",
    "E09000026","E09000027","E09000028","E09000029","E09000030",
    "E09000031","E09000032","E09000033",
}

# 读取 IMD Score + 人口分母，只保留 LSOA 代码、Score、人口
imd = pd.read_csv(IMD_SCORE_FILE, usecols=[
    "LSOA code (2021)",
    "Index of Multiple Deprivation (IMD) Score",
    "Total population: mid 2022",
])
imd.columns = ["LSOA21CD", "IMD_Score", "Population"]

# 读取 lookup，去重得到唯一 LSOA→MSOA 映射
lookup = pd.read_csv(LOOKUP_FILE, usecols=["LSOA21CD", "MSOA21CD", "MSOA21NM", "LAD22CD"])
lookup = lookup.drop_duplicates(subset=["LSOA21CD"])

# 合并 IMD Score 与 lookup
merged = imd.merge(lookup, on="LSOA21CD", how="left")

# 筛选伦敦
london = merged[merged["LAD22CD"].isin(LONDON_LAD_CODES)].copy()

# 人口加权平均：Σ(Score_i × Pop_i) / Σ(Pop_i)
london["Score_x_Pop"] = london["IMD_Score"] * london["Population"]

msoa_imd = (
    london.groupby(["MSOA21CD", "MSOA21NM"])
    .agg(
        Score_x_Pop_Sum=("Score_x_Pop", "sum"),
        Population_Total=("Population", "sum"),
        LSOA_Count=("LSOA21CD", "count"),
    )
    .reset_index()
)
msoa_imd["IMD_Score_PopWeighted"] = (
    msoa_imd["Score_x_Pop_Sum"] / msoa_imd["Population_Total"]
).round(4)
msoa_imd = msoa_imd.drop(columns="Score_x_Pop_Sum")
msoa_imd = msoa_imd[["MSOA21CD", "MSOA21NM", "IMD_Score_PopWeighted", "Population_Total", "LSOA_Count"]]

msoa_imd.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("完成！")
print(f"  伦敦 LSOA 数:  {len(london):,}")
print(f"  伦敦 MSOA 数:  {len(msoa_imd):,}")
print(f"  输出文件:      {OUTPUT_FILE}")
print(msoa_imd.head())
