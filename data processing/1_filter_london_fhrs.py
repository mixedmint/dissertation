import pandas as pd

LONDON_BOROUGHS = {
    "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley",
    "Camden", "City of London Corporation", "Croydon", "Ealing", "Enfield",
    "Greenwich", "Hackney", "Hammersmith and Fulham", "Haringey", "Harrow",
    "Havering", "Hillingdon", "Hounslow", "Islington", "Kensington and Chelsea",
    "Kingston-Upon-Thames", "Lambeth", "Lewisham", "Merton", "Newham",
    "Redbridge", "Richmond-Upon-Thames", "Southwark", "Sutton", "Tower Hamlets",
    "Waltham Forest", "Wandsworth", "Westminster"
}

print("正在读取 FHRS_All_en-GB.csv ...")
df = pd.read_csv("0_raw/FHRS_All_en-GB.csv", encoding="utf-8-sig")

london_df = df[df["LocalAuthorityName"].isin(LONDON_BOROUGHS)].copy()

before_drop = len(london_df)
london_df = london_df.dropna(subset=["Latitude", "Longitude"])

drop_cols = [
    "AddressLine1", "AddressLine2", "AddressLine3", "AddressLine4",
    "BusinessName", "NewRatingPending", "PostCode", "RatingKey",
    "RightToReply", "SchemeType", "Structural", "ConfidenceInManagement",
    "RatingDate"
]
london_df = london_df.drop(columns=[c for c in drop_cols if c in london_df.columns])

invalid_ratings = {"AwaitingInspection", "Exempt", "AwaitingPublication", "Pass"}
london_df = london_df[~london_df["RatingValue"].isin(invalid_ratings)]

before_dedup = len(london_df)
london_df = london_df.drop_duplicates(subset=["FHRSID"])

exclude_types = {
    "Distributors/Transporters",
    "Farmers/growers",
    "Hospitals/Childcare/Caring Premises",
    "Importers/Exporters",
    "Manufacturers/packers",
    "Mobile caterer",
    "School/college/university",
}
before_biztype = len(london_df)
london_df = london_df[~london_df["BusinessType"].isin(exclude_types)]

output_file = "1_FHRS_London.csv"
london_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"完成！")
print(f"  全国总行数:             {len(df):,}")
print(f"  伦敦筛选行数:           {before_drop:,}")
print(f"  删除无坐标行数:         {before_drop - before_dedup:,}")
print(f"  删除重复行数:           {before_dedup - before_biztype:,}")
print(f"  删除非餐饮业务类型行数: {before_biztype - len(london_df):,}")
print(f"  最终保留行数:           {len(london_df):,}")
print(f"  覆盖区数:               {london_df['LocalAuthorityName'].nunique()} / 33")
print(f"  输出文件:               {output_file}")
