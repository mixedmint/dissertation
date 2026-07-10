library(mgwnbr)

# ── 读取数据 ──
df <- read.csv("10_regression_with_coords.csv", fileEncoding = "UTF-8-BOM")
names(df)[names(df) == "X0.4"]    <- "age_0_4"
names(df)[names(df) == "X65plus"] <- "age_65plus"
df$log_restaurant_count <- log(df$restaurant_count)
df$log_pop_density      <- log(df$Population_density)

vars <- c("low_rating_count", "log_restaurant_count", "IMD",
          "Asian", "Black", "Other",
          "retailers___other", "takeaway_sandwich_shop",
          "pub_bar_nightclub", "supermarkets_hypermarkets",
          "hotel_bed_and_breakfast_guest_house", "other_catering_premises",
          "log_pop_density", "x", "y")
df <- df[complete.cases(df[, vars]), ]
df <- df[!apply(df[, vars], 1, function(r) any(is.infinite(r) | is.nan(r))), ]
cat("样本量:", nrow(df), "\n\n")

formula_gwnbr <- low_rating_count ~ IMD +
  Asian + Black + Other +
  retailers___other + takeaway_sandwich_shop +
  pub_bar_nightclub + supermarkets_hypermarkets +
  hotel_bed_and_breakfast_guest_house + other_catering_premises +
  log_pop_density

# ── 测试不同带宽 ──
bandwidths <- c(150, 200, 250)
results_list <- list()

for (h in bandwidths) {
  cat("正在跑 h =", h, "...\n")
  res <- suppressWarnings(mgwnbr(
    data           = df,
    formula        = formula_gwnbr,
    long           = "x",
    lat            = "y",
    band_method    = "adaptive_bsq",
    band_criterion = "aic",
    distribution   = "negbin",
    multiscale     = FALSE,
    distancekm     = FALSE,
    offset         = "log_restaurant_count",
    id             = "MSOA21CD",
    h              = h
  ))
  measures <- as.data.frame(t(res$measures))
  measures$h <- h
  results_list[[as.character(h)]] <- measures
  cat("  完成，AIC =", measures$AIC, "\n")
}

# ── 汇总比较 ──
comparison <- do.call(rbind, results_list)
comparison <- comparison[, c("h", setdiff(names(comparison), "h"))]
rownames(comparison) <- NULL

cat("\n── 带宽比较结果 ──\n")
print(comparison)

write.csv(comparison, "11_gwnbr_bw_comparison.csv",
          row.names = FALSE, fileEncoding = "UTF-8")
cat("\n结果已保存：11_gwnbr_bw_comparison.csv\n")
