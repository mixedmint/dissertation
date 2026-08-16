import pandas as pd
import geopandas as gpd

# 读取回归数据
df = pd.read_csv("data processing/7_regression.csv", encoding="utf-8-sig")

# 从 MSOA shapefile 提取质心坐标（EPSG:27700，单位：米）
msoa = gpd.read_file("0_raw/2021 London MSOA/2021_London_MSOA.shp")[["MSOA21CD", "geometry"]]
msoa["x"] = msoa.geometry.centroid.x
msoa["y"] = msoa.geometry.centroid.y
msoa = msoa[["MSOA21CD", "x", "y"]]

# 合并坐标
df = df.merge(msoa, on="MSOA21CD", how="left")

no_coord = df[["x", "y"]].isna().any(axis=1).sum()
if no_coord:
    print(f"警告：{no_coord} 个 MSOA 未匹配到坐标，将被删除")
    df = df.dropna(subset=["x", "y"])

df.to_csv("GWNBR/10_regression_with_coords.csv", index=False, encoding="utf-8-sig")

print("完成！")
print(f"  输出行数: {len(df):,}")
print(f"  输出文件: GWNBR/10_regression_with_coords.csv")
print(f"  列数:     {len(df.columns)}（含 x, y 坐标）")
