import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

FHRS_FILE = "1_FHRS_London.csv"
LSOA_SHP  = "0_raw/2021 London LSOA/2021_London_LSOA.shp"
MSOA_SHP  = "0_raw/2021 London MSOA/2021_London_MSOA.shp"

# 读取 FHRS，转成点几何（WGS84），再投影到 BNG
df = pd.read_csv(FHRS_FILE, encoding="utf-8-sig")
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
    crs="EPSG:4326"
).to_crs("EPSG:27700")

# ── LSOA 空间连接 ──
lsoa = gpd.read_file(LSOA_SHP)[["LSOA21CD", "LSOA21NM", "geometry"]]
joined_lsoa = gpd.sjoin(gdf, lsoa, how="left", predicate="within")

no_match_lsoa = joined_lsoa["LSOA21CD"].isna().sum()
if no_match_lsoa:
    print(f"  警告：{no_match_lsoa} 家餐厅未匹配到 LSOA，将被丢弃")

result_lsoa = joined_lsoa.dropna(subset=["LSOA21CD"]).drop(
    columns=["geometry", "index_right"]
)
result_lsoa.to_csv("3_FHRS_LSOA.csv", index=False, encoding="utf-8-sig")

# ── MSOA 空间连接 ──
msoa = gpd.read_file(MSOA_SHP)[["MSOA21CD", "MSOA21NM", "geometry"]]
joined_msoa = gpd.sjoin(gdf, msoa, how="left", predicate="within")

no_match_msoa = joined_msoa["MSOA21CD"].isna().sum()
if no_match_msoa:
    print(f"  警告：{no_match_msoa} 家餐厅未匹配到 MSOA，将被丢弃")

result_msoa = joined_msoa.dropna(subset=["MSOA21CD"]).drop(
    columns=["geometry", "index_right"]
)
result_msoa.to_csv("3_FHRS_MSOA.csv", index=False, encoding="utf-8-sig")

print("完成！")
print(f"  FHRS 原始行数:      {len(df):,}")
print(f"  2_FHRS_LSOA.csv:   {len(result_lsoa):,} 行，{result_lsoa['LSOA21CD'].nunique():,} 个 LSOA")
print(f"  2_FHRS_MSOA.csv:   {len(result_msoa):,} 行，{result_msoa['MSOA21CD'].nunique():,} 个 MSOA")
