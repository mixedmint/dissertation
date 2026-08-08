library(mgwnbr)

# ── 1. 读取数据 ──
df <- read.csv("10_regression_with_coords.csv", fileEncoding = "UTF-8-BOM")

# 处理列名
names(df)[names(df) == "X0.4"]    <- "age_0_4"
names(df)[names(df) == "X65plus"] <- "age_65plus"

# log 变换
df$log_restaurant_count <- log(df$restaurant_count)
df$log_pop_density      <- log(df$Population_density)

# 删除缺失值和 Inf
vars <- c("low_rating_count", "log_restaurant_count", "IMD",
          "Asian", "Other",
          "retailers___other", "takeaway_sandwich_shop",
          "pub_bar_nightclub", "supermarkets_hypermarkets",
          "hotel_bed_and_breakfast_guest_house", "other_catering_premises",
          "log_pop_density", "x", "y")
df <- df[complete.cases(df[, vars]), ]
df <- df[!apply(df[, vars], 1, function(r) any(is.infinite(r) | is.nan(r))), ]
cat("样本量:", nrow(df), "\n\n")

# ── 2. 运行 GWNBR ──
# multiscale=FALSE：标准 GWNBR（单一带宽）
# offset：log(restaurant_count) 列名
# distancekm=FALSE：坐标为 BNG 米制，不转换
cat("正在运行 GWNBR（可能需要数分钟）...\n")
result <- suppressWarnings(mgwnbr(
  data           = df,
  formula        = low_rating_count ~ IMD +
                     Asian + Other +
                     retailers___other + takeaway_sandwich_shop +
                     pub_bar_nightclub + supermarkets_hypermarkets +
                     hotel_bed_and_breakfast_guest_house + other_catering_premises +
                     log_pop_density,
  long           = "x",
  lat            = "y",
  band_method    = "adaptive_bsq",
  band_criterion = "aic",
  distribution   = "negbin",
  multiscale     = FALSE,
  distancekm     = FALSE,
  offset         = "log_restaurant_count",
  id             = "MSOA21CD",
  h              = 140
))

# ── 3. 查看结果 ──
cat("\n── 带宽 ──\n")
print(result$general_bandwidth)

bw_df <- data.frame(bandwidth = result$general_bandwidth, n = nrow(df))
write.csv(bw_df, "11_gwnbr_bandwidth.csv", row.names = FALSE, fileEncoding = "UTF-8")

cat("\n── 模型拟合指标 ──\n")
print(result$measures)

cat("\n── 全局系数 ──\n")
print(result$global_param_estimates)

# ── 4. 保存全局系数和拟合指标 ──
global_coefs <- as.data.frame(result$global_param_estimates)
write.csv(global_coefs, "11_gwnbr_global_coefs.csv",
          row.names = TRUE, fileEncoding = "UTF-8")

measures <- as.data.frame(result$measures)
global_measures <- as.data.frame(result$global_measures)
write.csv(measures,        "11_gwnbr_measures.csv",        row.names = TRUE, fileEncoding = "UTF-8")
write.csv(global_measures, "11_gwnbr_global_measures.csv", row.names = TRUE, fileEncoding = "UTF-8")

cat("\n── 全局模型拟合指标 ──\n")
print(result$global_measures)

# ── 5. 保存局部系数 ──
local_coefs <- as.data.frame(result$mgwr_param_estimates)
local_coefs$MSOA21CD <- df$MSOA21CD
local_coefs$MSOA21NM <- df$MSOA21NM
local_coefs$x        <- df$x
local_coefs$y        <- df$y

write.csv(local_coefs, "11_gwnbr_local_coefs.csv",
          row.names = FALSE, fileEncoding = "UTF-8")

# 局部标准误
local_se <- as.data.frame(result$mgwr_se)
local_se$MSOA21CD <- df$MSOA21CD
write.csv(local_se, "11_gwnbr_local_se.csv",
          row.names = FALSE, fileEncoding = "UTF-8")

cat("\n结果已保存：\n")
cat("  11_gwnbr_global_coefs.csv（全局系数）\n")
cat("  11_gwnbr_measures.csv（GWNBR 拟合指标）\n")
cat("  11_gwnbr_global_measures.csv（全局模型拟合指标）\n")
cat("  11_gwnbr_local_coefs.csv（局部系数）\n")
cat("  11_gwnbr_local_se.csv（局部标准误）\n")
cat("  11_gwnbr_bandwidth.csv（实际使用带宽）\n")

