# =============================================================================
# ultimate_loss_prediction.R
#
# Ultimate Incurred Claim Cost Prediction
# Workers' Compensation Insurance — Actuarial NLP + Machine Learning Pipeline
#
# Author  : Actuarial Data Science Research
# Date    : 2024
# R Version: 4.3+
#
# Description:
#   End-to-end actuarial ML pipeline predicting UltimateIncurredClaimCost
#   from workers' compensation claims data. Combines structured claim
#   variables with BERT-based text embeddings from ClaimDescription.
#
# Pipeline:
#   1.  Environment setup & package installation
#   2.  Global configuration (seed, paths, parameters)
#   3.  Data loading & initial inspection
#   4.  Data leakage audit
#   5.  Feature engineering & preprocessing
#   6.  BERT sentence embeddings (via reticulate / Python)
#   7.  UMAP dimensionality reduction
#   8.  Unsupervised text clustering (K-Means + HDBSCAN)
#   9.  Train / test split & feature matrix construction
#   10. XGBoost hyperparameter tuning (grid search + 5-fold CV)
#   11. Final model training
#   12. Model evaluation (RMSE, MAE, R²)
#   13. Diagnostic plots
#   14. SHAP explainability (global + per-claim)
#   15. Model persistence & session info
#
# Usage:
#   Rscript ultimate_loss_prediction.R
#   — or — open in RStudio and run interactively
#
# Input : data.csv  (set DATA_PATH in Section 2)
# Outputs:
#   xgb_model.json        — trained XGBoost model
#   predictions.csv       — test-set predictions
#   encoding_maps.rds     — categorical encoding maps
#   clustering_model.rds  — K-Means + UMAP artefacts
#   plots/                — all diagnostic plots (PNG)
# =============================================================================


# =============================================================================
# SECTION 1 — PACKAGE INSTALLATION & LOADING
# =============================================================================
# We install only packages that are not already present, then load them all.
# This makes the script safe to re-run without reinstalling everything.

cat("\n── Section 1: Installing & Loading Packages ──\n")

install_if_missing <- function(pkgs) {
  missing <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
  if (length(missing) > 0) {
    message("Installing: ", paste(missing, collapse = ", "))
    install.packages(missing, repos = "https://cran.rstudio.com/", quiet = TRUE)
  }
}

r_packages <- c(
  # ── Core data manipulation ─────────────────────────────────────────────────
  "tidyverse",      # dplyr, ggplot2, tidyr, stringr, purrr, readr
  "data.table",     # fast CSV reading
  "lubridate",      # date/time arithmetic

  # ── Text processing ────────────────────────────────────────────────────────
  "text2vec",       # TF-IDF / GloVe baseline utilities
  "reticulate",     # Python interop for BERT via sentence-transformers

  # ── Clustering ─────────────────────────────────────────────────────────────
  "cluster",        # silhouette scores
  "factoextra",     # cluster visualisation helpers
  "umap",           # UMAP (R implementation — used for post-hoc plots)

  # ── Modelling ──────────────────────────────────────────────────────────────
  "xgboost",        # gradient boosted trees
  "Matrix",         # sparse matrix support
  "caret",          # stratified splitting, CV helpers

  # ── Explainability ─────────────────────────────────────────────────────────
  "shapviz",        # SHAP waterfall, beeswarm, dependence, force plots
  "SHAPforxgboost", # TreeSHAP computation & ggplot-ready long format

  # ── Visualisation helpers ──────────────────────────────────────────────────
  "scales",         # dollar/comma/percent axis labels
  "gridExtra",      # multi-panel plot layouts
  "corrplot",       # correlation heatmap
  "viridis"         # perceptually uniform colour scales
)

install_if_missing(r_packages)

suppressPackageStartupMessages(
  invisible(lapply(r_packages, library, character.only = TRUE))
)

cat("✅ All R packages loaded.\n")


# =============================================================================
# SECTION 2 — GLOBAL CONFIGURATION
# =============================================================================
# Centralise all configuration so the script is easy to adapt.
# Change DATA_PATH to point to your CSV file.

cat("\n── Section 2: Global Configuration ──\n")

# Reproducibility seed (applied everywhere)
GLOBAL_SEED <- 42L
set.seed(GLOBAL_SEED)

# All tunable parameters in one list
CONFIG <- list(
  # ── Paths ──────────────────────────────────────────────────────────────────
  data_path       = "data.csv",            # input data
  output_dir      = "plots",              # directory for saved plots
  model_path      = "xgb_model.json",     # XGBoost model output
  pred_path       = "predictions.csv",    # test-set predictions output

  # ── Modelling ──────────────────────────────────────────────────────────────
  train_frac      = 0.80,                 # train/test split fraction
  cv_folds        = 5L,                   # k-fold CV
  early_stop      = 30L,                  # early stopping patience (rounds)
  max_nrounds     = 600L,                 # maximum boosting iterations

  # ── SHAP ───────────────────────────────────────────────────────────────────
  shap_n_sample   = 5000L,                # observations for SHAP (speed/accuracy)

  # ── NLP / Clustering ───────────────────────────────────────────────────────
  bert_model      = "all-MiniLM-L6-v2",   # sentence-transformers model name
  bert_batch      = 256L,                 # BERT encoding batch size
  umap_2d_nn      = 15L,                  # UMAP 2D: n_neighbors
  umap_10d_nn     = 15L,                  # UMAP 10D: n_neighbors
  kmeans_k_max    = 12L                   # max K to search in elbow method
)

# Create output directory for plots
if (!dir.exists(CONFIG$output_dir)) dir.create(CONFIG$output_dir)

# ── ggplot2 theme applied globally ────────────────────────────────────────────
theme_actuarial <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title       = element_text(face = "bold", size = base_size + 2),
      plot.subtitle    = element_text(colour = "grey40", size = base_size - 1),
      plot.caption     = element_text(colour = "grey60", size = base_size - 3),
      panel.grid.minor = element_blank(),
      legend.position  = "bottom",
      strip.text       = element_text(face = "bold")
    )
}
theme_set(theme_actuarial())

# Helper: save a plot to the output directory
save_plot <- function(plot_obj, filename, w = 12, h = 7) {
  path <- file.path(CONFIG$output_dir, filename)
  ggsave(path, plot = plot_obj, width = w, height = h, dpi = 150)
  cat(sprintf("   Saved: %s\n", path))
}

cat("✅ Configuration set. Seed:", GLOBAL_SEED, "\n")
cat("   Data path:", CONFIG$data_path, "\n")
cat("   Plot dir: ", CONFIG$output_dir, "\n")


# =============================================================================
# SECTION 3 — DATA LOADING & INITIAL INSPECTION
# =============================================================================
# Load the raw data and perform a structured EDA before any transformations.

cat("\n── Section 3: Data Loading & EDA ──\n")

# data.table::fread() is significantly faster than read.csv() for large files
df_raw <- data.table::fread(CONFIG$data_path) |> as_tibble()

cat(sprintf("Dataset loaded: %s rows × %d columns\n",
            scales::comma(nrow(df_raw)), ncol(df_raw)))

# ── Column types ───────────────────────────────────────────────────────────────
cat("\nColumn types:\n")
glimpse(df_raw)

# ── Summary statistics ─────────────────────────────────────────────────────────
cat("\nSummary statistics:\n")
print(summary(df_raw))

# ── Categorical distributions ──────────────────────────────────────────────────
cat("\nGender distribution:\n");         print(table(df_raw$Gender,        useNA = "ifany"))
cat("\nMaritalStatus distribution:\n");  print(table(df_raw$MaritalStatus, useNA = "ifany"))
cat("\nPartTimeFullTime distribution:\n"); print(table(df_raw$PartTimeFullTime, useNA = "ifany"))

# ── Missing value analysis ─────────────────────────────────────────────────────
missing_tbl <- df_raw |>
  summarise(across(everything(), ~ sum(is.na(.)))) |>
  pivot_longer(everything(), names_to = "column", values_to = "n_missing") |>
  mutate(pct_missing = round(n_missing / nrow(df_raw) * 100, 2)) |>
  arrange(desc(n_missing))

cat("\nMissing value summary:\n")
print(missing_tbl)

# Plot missing values (only if any exist)
missing_plot_data <- missing_tbl |> filter(n_missing > 0)
if (nrow(missing_plot_data) > 0) {
  p_missing <- ggplot(missing_plot_data,
                      aes(x = reorder(column, -pct_missing),
                          y = pct_missing, fill = pct_missing)) +
    geom_col(width = 0.7) +
    geom_text(aes(label = paste0(pct_missing, "%")),
              vjust = -0.4, size = 3.5) +
    scale_fill_gradient(low = "#FFF176", high = "#C62828", guide = "none") +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    labs(title    = "Missing Value Analysis",
         subtitle = "Columns with at least one missing observation",
         x = NULL, y = "% Missing") +
    theme(axis.text.x = element_text(angle = 40, hjust = 1))
  save_plot(p_missing, "01_missing_values.png")
}

# ── Target variable distribution ───────────────────────────────────────────────
# Claim costs are right-skewed — log transform improves model fit
target_stats <- df_raw |>
  summarise(
    n      = n(),
    mean   = mean(UltimateIncurredClaimCost, na.rm = TRUE),
    median = median(UltimateIncurredClaimCost, na.rm = TRUE),
    sd     = sd(UltimateIncurredClaimCost, na.rm = TRUE),
    p95    = quantile(UltimateIncurredClaimCost, 0.95, na.rm = TRUE),
    p99    = quantile(UltimateIncurredClaimCost, 0.99, na.rm = TRUE),
    max    = max(UltimateIncurredClaimCost, na.rm = TRUE)
  )

cat("\nTarget variable statistics:\n")
cat(sprintf("  Mean:   $%s\n", scales::comma(round(target_stats$mean,   0))))
cat(sprintf("  Median: $%s\n", scales::comma(round(target_stats$median, 0))))
cat(sprintf("  P95:    $%s\n", scales::comma(round(target_stats$p95,    0))))
cat(sprintf("  P99:    $%s\n", scales::comma(round(target_stats$p99,    0))))
cat(sprintf("  Max:    $%s\n", scales::comma(round(target_stats$max,    0))))

p_raw <- ggplot(df_raw, aes(x = UltimateIncurredClaimCost)) +
  geom_histogram(bins = 100, fill = "#1565C0", alpha = 0.85, colour = "white") +
  geom_vline(xintercept = target_stats$mean,   colour = "#E53935", linetype = "dashed") +
  geom_vline(xintercept = target_stats$median, colour = "#2E7D32", linetype = "dotted") +
  scale_x_continuous(labels = scales::dollar_format(scale = 1e-3, suffix = "K")) +
  labs(title    = "Raw Target Distribution",
       subtitle = "Red dashed = mean | Green dotted = median",
       x = "Ultimate Claim Cost", y = "Count")

p_log <- ggplot(df_raw, aes(x = log1p(UltimateIncurredClaimCost))) +
  geom_histogram(bins = 80, fill = "#2E7D32", alpha = 0.85, colour = "white") +
  geom_density(aes(y = after_stat(count)), colour = "#E53935", linewidth = 0.8) +
  labs(title    = "Log-Transformed Target",
       subtitle = "log1p(x) stabilises variance",
       x = "log1p(UltimateIncurredClaimCost)", y = "Count")

p_target_combined <- gridExtra::arrangeGrob(p_raw, p_log, nrow = 1)
save_plot(p_target_combined, "02_target_distribution.png")


# =============================================================================
# SECTION 4 — DATA LEAKAGE AUDIT
# =============================================================================
# Check all features for potential target leakage before any modelling.
# Rule: correlation |r| > 0.95 with the target is a strong leakage signal.

cat("\n── Section 4: Leakage Audit ──\n")

leakage_check <- df_raw |>
  select(where(is.numeric), -UltimateIncurredClaimCost) |>
  cor(df_raw$UltimateIncurredClaimCost, use = "complete.obs") |>
  as.data.frame() |>
  rownames_to_column("feature") |>
  rename(corr_with_target = V1) |>
  arrange(desc(abs(corr_with_target)))

cat("Pearson correlation with target:\n")
print(leakage_check)

high_corr <- leakage_check |> filter(abs(corr_with_target) > 0.95)
if (nrow(high_corr) > 0) {
  cat("\n⚠️  WARNING: Potential leakage features (|r| > 0.95):\n")
  print(high_corr)
} else {
  cat("\n✅ No extreme correlations (|r| > 0.95). Leakage risk is low.\n")
}
# NOTE: InitialIncurredCalimsCost is an early reserve estimate — included
# because it is known at claim intake. Remove it for pure pricing models
# where no initial estimate is available.


# =============================================================================
# SECTION 5 — FEATURE ENGINEERING & PREPROCESSING
# =============================================================================

cat("\n── Section 5: Feature Engineering & Preprocessing ──\n")

# ── 5.1 Date parsing & temporal features ──────────────────────────────────────
# All temporal features are derived from timestamps available at claim intake.
df <- df_raw |>
  mutate(
    dt_accident  = lubridate::ymd_hms(DateTimeOfAccident, quiet = TRUE),
    dt_reported  = lubridate::ymd_hms(DateReported,       quiet = TRUE),

    # Reporting delay (days): IBNR proxy — long delays indicate complex claims
    ReportingDelay   = as.numeric(difftime(dt_reported, dt_accident, units = "days")),

    # Development year: captures trend/IBNER effects
    AccidentYear     = lubridate::year(dt_accident),

    # Month: seasonal patterns (construction, agriculture)
    AccidentMonth    = lubridate::month(dt_accident),

    # Weekday (1=Sun … 7=Sat): Monday Effect — contested claims filed Monday
    AccidentWeekday  = lubridate::wday(dt_accident, label = FALSE),

    # Hour: shift-related injury patterns (early morning, overnight)
    AccidentHour     = lubridate::hour(dt_accident),

    # Weekend flag: binary — weekend accidents have different characteristics
    IsWeekend        = as.integer(AccidentWeekday %in% c(1L, 7L))
  )

cat(sprintf("Accident years: %d – %d\n",
            min(df$AccidentYear, na.rm = TRUE),
            max(df$AccidentYear, na.rm = TRUE)))
cat(sprintf("Reporting delay: %.0f to %.0f days (median: %.0f)\n",
            min(df$ReportingDelay, na.rm = TRUE),
            max(df$ReportingDelay, na.rm = TRUE),
            median(df$ReportingDelay, na.rm = TRUE)))

# Plot reporting delay distribution
p_delay_dist <- df |>
  filter(ReportingDelay >= 0, ReportingDelay <= 365) |>
  ggplot(aes(x = ReportingDelay)) +
  geom_histogram(bins = 60, fill = "#5C6BC0", alpha = 0.85, colour = "white") +
  geom_vline(xintercept = median(df$ReportingDelay, na.rm = TRUE),
             colour = "#E53935", linetype = "dashed", linewidth = 0.9) +
  labs(title    = "Reporting Delay Distribution",
       subtitle = "Red dashed = median | Capped at 365 days for display",
       x = "Days between accident and report", y = "Count")
save_plot(p_delay_dist, "03_reporting_delay.png")

# ── 5.2 Missing value imputation ───────────────────────────────────────────────
# LEAKAGE NOTE: These medians are computed from the full dataset here.
# In production (Section 9), they are recomputed from training data only.
impute_vals <- list(
  Age                = median(df$Age,               na.rm = TRUE),
  WeeklyWages        = median(df$WeeklyWages,        na.rm = TRUE),
  HoursWorkedPerWeek = median(df$HoursWorkedPerWeek, na.rm = TRUE),
  DaysWorkedPerWeek  = median(df$DaysWorkedPerWeek,  na.rm = TRUE),
  ReportingDelay     = median(df$ReportingDelay,     na.rm = TRUE),
  InitialCost        = median(df$InitialIncurredCalimsCost, na.rm = TRUE)
)

df <- df |>
  mutate(
    # Numeric: median imputation (robust to outliers)
    Age                       = coalesce(Age,                impute_vals$Age),
    WeeklyWages               = coalesce(WeeklyWages,        impute_vals$WeeklyWages),
    HoursWorkedPerWeek        = coalesce(HoursWorkedPerWeek, impute_vals$HoursWorkedPerWeek),
    DaysWorkedPerWeek         = coalesce(DaysWorkedPerWeek,  impute_vals$DaysWorkedPerWeek),
    ReportingDelay            = coalesce(ReportingDelay,     impute_vals$ReportingDelay),
    InitialIncurredCalimsCost = coalesce(InitialIncurredCalimsCost, impute_vals$InitialCost),

    # Categorical: 'Unknown' category preserves missingness signal
    Gender           = coalesce(Gender,           "Unknown"),
    MaritalStatus    = coalesce(MaritalStatus,    "Unknown"),
    PartTimeFullTime = coalesce(PartTimeFullTime, "Unknown"),

    # Text: empty string → BERT produces a near-zero embedding
    ClaimDescription = coalesce(ClaimDescription, "")
  )

cat("✅ Imputation complete.\n")

# ── 5.3 Categorical encoding ───────────────────────────────────────────────────
# Integer encoding for XGBoost (handles ordinal splits natively)
gender_levels   <- sort(unique(df$Gender))
marital_levels  <- sort(unique(df$MaritalStatus))
parttime_levels <- sort(unique(df$PartTimeFullTime))

df <- df |>
  mutate(
    Gender_enc           = as.integer(factor(Gender,           levels = gender_levels)),
    MaritalStatus_enc    = as.integer(factor(MaritalStatus,    levels = marital_levels)),
    PartTimeFullTime_enc = as.integer(factor(PartTimeFullTime, levels = parttime_levels))
  )

# Save encoding maps for future scoring of new data
encoding_maps <- list(
  Gender           = setNames(seq_along(gender_levels),   gender_levels),
  MaritalStatus    = setNames(seq_along(marital_levels),  marital_levels),
  PartTimeFullTime = setNames(seq_along(parttime_levels), parttime_levels)
)

cat("Encoding maps:\n")
purrr::iwalk(encoding_maps, function(m, nm) {
  cat(sprintf("  %s: %s\n", nm, paste(names(m), "→", m, collapse = ", "))  )
})

# ── 5.4 Log-transform skewed continuous features ───────────────────────────────
# log1p(x) = log(1+x): handles x=0 and is inverted by expm1()
df <- df |>
  mutate(
    log_WeeklyWages = log1p(WeeklyWages),
    log_InitialCost = log1p(InitialIncurredCalimsCost),
    log_Target      = log1p(UltimateIncurredClaimCost)   # model target
  )

cat("✅ Encoding and log-transforms complete.\n")

# ── 5.5 Correlation heatmap ────────────────────────────────────────────────────
num_for_corr <- df |>
  select(Age, log_WeeklyWages, HoursWorkedPerWeek, DaysWorkedPerWeek,
         DependentChildren, DependentsOther, ReportingDelay,
         AccidentMonth, AccidentWeekday, AccidentHour, AccidentYear,
         log_InitialCost, log_Target)

cor_mat <- cor(num_for_corr, use = "complete.obs")

png(file.path(CONFIG$output_dir, "04_correlation_heatmap.png"),
    width = 1400, height = 1200, res = 130)
corrplot::corrplot(
  cor_mat,
  method      = "color", type = "upper",
  tl.cex      = 0.80, tl.col = "black",
  col         = corrplot::COL2("RdBu", 200),
  addCoef.col = "black", number.cex = 0.55,
  title       = "Correlation Matrix — Numeric Features",
  mar         = c(0, 0, 2, 0)
)
invisible(dev.off())
cat("   Saved: plots/04_correlation_heatmap.png\n")

# ── 5.6 Feature visualisations by group ───────────────────────────────────────
p_by_gender <- df |>
  filter(Gender %in% c("M", "F")) |>
  ggplot(aes(x = Gender, y = log_Target, fill = Gender)) +
  geom_boxplot(alpha = 0.7, outlier.size = 0.5) +
  scale_fill_manual(values = c("F" = "#E91E63", "M" = "#1565C0")) +
  labs(title = "Claim Cost by Gender", x = NULL, y = "log1p(Claim Cost)") +
  theme(legend.position = "none")

p_by_ptype <- df |>
  filter(PartTimeFullTime %in% c("F", "P")) |>
  ggplot(aes(x = PartTimeFullTime, y = log_Target, fill = PartTimeFullTime)) +
  geom_violin(alpha = 0.6) +
  geom_boxplot(width = 0.15, fill = "white", outlier.size = 0.3) +
  scale_fill_manual(values = c("F" = "#2E7D32", "P" = "#F57F17")) +
  labs(title = "Full vs Part Time", x = NULL, y = "log1p(Claim Cost)") +
  theme(legend.position = "none")

p_by_month <- df |>
  group_by(AccidentMonth) |>
  summarise(median_cost = median(UltimateIncurredClaimCost), .groups = "drop") |>
  ggplot(aes(x = factor(AccidentMonth), y = median_cost)) +
  geom_col(fill = "#5C6BC0", alpha = 0.85) +
  scale_y_continuous(labels = scales::dollar_format()) +
  scale_x_discrete(labels = month.abb) +
  labs(title = "Median Cost by Month", x = "Month", y = "Median Cost")

p_delay_cost <- df |>
  filter(ReportingDelay >= 0, ReportingDelay <= 180) |>
  ggplot(aes(x = ReportingDelay, y = log_Target)) +
  geom_hex(bins = 50) +
  scale_fill_viridis_c(option = "plasma") +
  labs(title = "Cost vs Reporting Delay",
       x = "Delay (days)", y = "log1p(Claim Cost)", fill = "Count")

p_eda_grid <- gridExtra::arrangeGrob(
  p_by_gender, p_by_ptype, p_by_month, p_delay_cost, nrow = 2
)
save_plot(p_eda_grid, "05_feature_distributions.png", w = 14, h = 9)


# =============================================================================
# SECTION 6 — BERT SENTENCE EMBEDDINGS
# =============================================================================
# Generate 384-dimensional embeddings from ClaimDescription using
# all-MiniLM-L6-v2 via Python's sentence-transformers library.

cat("\n── Section 6: BERT Embeddings ──\n")

# ── 6.1 Python environment setup ──────────────────────────────────────────────
library(reticulate)
use_python("/usr/bin/python3", required = TRUE)

# Install required Python packages (quiet = suppress pip output)
system("pip install -q sentence-transformers hdbscan umap-learn scikit-learn")
cat("Python packages ready.\n")

# ── 6.2 Prepare texts for embedding ───────────────────────────────────────────
# Minimal cleaning: BERT tokenisation handles most normalisation internally
claim_texts <- df$ClaimDescription |>
  stringr::str_to_upper() |>
  stringr::str_squish() |>
  stringr::str_replace_all("[^[:print:]]", " ") |>
  replace_na("")

# Export to Python namespace via reticulate
py$claim_texts  <- claim_texts
py$bert_model   <- CONFIG$bert_model
py$bert_batch   <- CONFIG$bert_batch
py$global_seed  <- GLOBAL_SEED

cat(sprintf("Encoding %s claim descriptions...\n",
            scales::comma(length(claim_texts))))
cat(sprintf("Sample: %s\n", claim_texts[1]))

# ── 6.3 Generate embeddings in Python ─────────────────────────────────────────
py_run_string("
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Select GPU if available (significant speedup for 54K texts)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Load pre-trained model: 6-layer distilled BERT, 384-dim, 22M params
# Trained on 1B sentence pairs via contrastive learning
encoder = SentenceTransformer(bert_model, device=device)

# Encode all descriptions; normalize_embeddings=True → L2 unit norm
# This makes cosine similarity equal to dot product (efficient)
embeddings = encoder.encode(
    claim_texts,
    batch_size           = bert_batch,
    show_progress_bar    = True,
    normalize_embeddings = True,
    device               = device
)

print(f'Embedding matrix: {embeddings.shape[0]:,} x {embeddings.shape[1]}')
print(f'Memory: {embeddings.nbytes / 1e6:.1f} MB')
print(f'Norm check (should be ~1.0): {np.linalg.norm(embeddings[0]):.6f}')
")

# Pull back to R as numeric matrix
bert_embeddings <- py$embeddings
cat(sprintf("✅ Embeddings in R: %d × %d matrix\n",
            nrow(bert_embeddings), ncol(bert_embeddings)))


# =============================================================================
# SECTION 7 — UMAP DIMENSIONALITY REDUCTION
# =============================================================================
# Two projections of the 384-dim BERT space:
#   • 2D  → visualisation (scatter plots)
#   • 10D → clustering (preserves more structure than 2D, less noise than 384D)

cat("\n── Section 7: UMAP Reduction ──\n")

py$embeddings_np  <- bert_embeddings
py$umap_2d_nn     <- CONFIG$umap_2d_nn
py$umap_10d_nn    <- CONFIG$umap_10d_nn

py_run_string("
import umap
import numpy as np

# 2D: for visualisation — min_dist=0.1 prevents over-crowding
print('UMAP 2D (visualisation)...')
reducer_2d = umap.UMAP(
    n_components = 2,
    n_neighbors  = umap_2d_nn,
    min_dist     = 0.1,
    metric       = 'cosine',
    random_state = global_seed,
    low_memory   = True
)
umap_2d = reducer_2d.fit_transform(embeddings_np)
print(f'  Done: {umap_2d.shape}')

# 10D: for clustering — min_dist=0.0 forces tighter cluster structure
print('UMAP 10D (clustering)...')
reducer_10d = umap.UMAP(
    n_components = 10,
    n_neighbors  = umap_10d_nn,
    min_dist     = 0.0,
    metric       = 'cosine',
    random_state = global_seed,
    low_memory   = True
)
umap_10d = reducer_10d.fit_transform(embeddings_np)
print(f'  Done: {umap_10d.shape}')
")

umap_2d  <- py$umap_2d
umap_10d <- py$umap_10d
cat(sprintf("✅ UMAP: 2D = %d×%d | 10D = %d×%d\n",
            nrow(umap_2d), ncol(umap_2d),
            nrow(umap_10d), ncol(umap_10d)))


# =============================================================================
# SECTION 8 — UNSUPERVISED TEXT CLUSTERING
# =============================================================================
# Cluster UMAP-10D embeddings to create a semantic injury-type feature.

cat("\n── Section 8: Text Clustering ──\n")

# ── 8.1 Optimal K: Elbow + Silhouette ─────────────────────────────────────────
# Silhouette is O(n²); use a subsample for speed
set.seed(GLOBAL_SEED)
K_MAX    <- CONFIG$kmeans_k_max
samp_n   <- 5000L
samp_idx <- sample(nrow(umap_10d), min(samp_n, nrow(umap_10d)))
umap_samp <- umap_10d[samp_idx, ]

cluster_metrics <- tibble(k = 2:K_MAX, wcss = NA_real_, silhouette = NA_real_)

cat(sprintf("Evaluating K = 2 to %d on %s-obs sample...\n",
            K_MAX, scales::comma(samp_n)))
cat(sprintf("%-6s %-14s %-12s\n", "K", "WCSS", "Silhouette"))
cat(strrep("-", 34), "\n")

for (k in 2:K_MAX) {
  km  <- kmeans(umap_samp, centers = k, nstart = 15L, iter.max = 150L)
  sil <- cluster::silhouette(km$cluster, dist(umap_samp))
  cluster_metrics$wcss[k - 1]       <- km$tot.withinss
  cluster_metrics$silhouette[k - 1] <- mean(sil[, 3])
  cat(sprintf("K=%-4d  WCSS=%-12.1f  Sil=%.4f\n",
              k, km$tot.withinss, mean(sil[, 3])))
}

optimal_k <- cluster_metrics$k[which.max(cluster_metrics$silhouette)]
best_sil  <- max(cluster_metrics$silhouette)
cat(sprintf("\n🏆 Optimal K = %d  (Silhouette = %.4f)\n", optimal_k, best_sil))

# Plot elbow + silhouette
p_elbow <- ggplot(cluster_metrics, aes(x = k, y = wcss)) +
  geom_line(colour = "#1565C0", linewidth = 0.9) +
  geom_point(colour = "#1565C0", size = 3) +
  geom_vline(xintercept = optimal_k, colour = "#E53935", linetype = "dashed") +
  scale_x_continuous(breaks = 2:K_MAX) +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "Elbow Method", subtitle = "Diminishing WCSS reduction",
       x = "K", y = "Total Within-Cluster SS")

p_sil_plot <- ggplot(cluster_metrics, aes(x = k, y = silhouette)) +
  geom_line(colour = "#2E7D32", linewidth = 0.9) +
  geom_point(colour = "#2E7D32", size = 3) +
  geom_vline(xintercept = optimal_k, colour = "#E53935", linetype = "dashed") +
  geom_point(data = cluster_metrics |> filter(k == optimal_k),
             colour = "#E53935", size = 5, shape = 21,
             fill = "#E53935", alpha = 0.5) +
  scale_x_continuous(breaks = 2:K_MAX) +
  labs(title = "Silhouette Analysis", subtitle = paste0("Optimal K = ", optimal_k),
       x = "K", y = "Average Silhouette Score")

p_cluster_search <- gridExtra::arrangeGrob(p_elbow, p_sil_plot, nrow = 1)
save_plot(p_cluster_search, "06_cluster_optimisation.png")

# ── 8.2 Final K-Means on full UMAP-10D ────────────────────────────────────────
# nstart=25: try 25 random initialisations, keep best solution
set.seed(GLOBAL_SEED)
km_final <- kmeans(umap_10d, centers = optimal_k, nstart = 25L, iter.max = 300L)
df$TextCluster <- as.integer(km_final$cluster)

cat("Cluster size distribution:\n")
print(table(df$TextCluster))

# ── 8.3 HDBSCAN (density-based alternative) ───────────────────────────────────
# Does not require specifying K; noise points labelled -1
py$umap_10d_np <- umap_10d

py_run_string("
import hdbscan
import numpy as np

print('Fitting HDBSCAN...')
clusterer = hdbscan.HDBSCAN(
    min_cluster_size     = 50,
    min_samples          = 5,
    metric               = 'euclidean',
    cluster_selection_method = 'eom',
    core_dist_n_jobs     = -1
)
hdbscan_labels = clusterer.fit_predict(umap_10d_np)
n_cl   = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
n_noise= int(np.sum(hdbscan_labels == -1))
print(f'  Clusters: {n_cl} | Noise points: {n_noise:,} ({100*n_noise/len(hdbscan_labels):.1f}%)')
")
df$TextCluster_HDBSCAN <- as.integer(py$hdbscan_labels)
cat("HDBSCAN distribution (-1=noise):\n")
print(table(df$TextCluster_HDBSCAN))

# ── 8.4 UMAP 2D cluster visualisations ────────────────────────────────────────
umap_vis <- tibble(
  UMAP1    = umap_2d[, 1],
  UMAP2    = umap_2d[, 2],
  KMeans   = factor(df$TextCluster),
  HDBSCAN  = factor(df$TextCluster_HDBSCAN),
  log_cost = df$log_Target
)

p_km_umap <- ggplot(umap_vis, aes(UMAP1, UMAP2, colour = KMeans)) +
  geom_point(alpha = 0.30, size = 0.5) +
  scale_colour_brewer(palette = "Set1") +
  guides(colour = guide_legend(override.aes = list(size = 3, alpha = 1))) +
  labs(title    = "K-Means Clusters on UMAP 2D",
       subtitle = paste0("K = ", optimal_k, " | BERT → UMAP 10D → K-Means"),
       colour = "Cluster")

p_cost_umap <- ggplot(umap_vis, aes(UMAP1, UMAP2, colour = log_cost)) +
  geom_point(alpha = 0.30, size = 0.5) +
  scale_colour_viridis_c(option = "plasma") +
  labs(title = "Claim Cost Gradient on UMAP 2D",
       colour = "log1p(Cost)")

p_umap_combined <- gridExtra::arrangeGrob(p_km_umap, p_cost_umap, nrow = 1)
save_plot(p_umap_combined, "07_umap_clusters.png", w = 14, h = 6)

# ── 8.5 Cluster cost profiling ─────────────────────────────────────────────────
cluster_profile <- df |>
  group_by(TextCluster) |>
  summarise(
    N          = n(),
    MedianCost = median(UltimateIncurredClaimCost),
    MeanCost   = mean(UltimateIncurredClaimCost),
    P90Cost    = quantile(UltimateIncurredClaimCost, 0.90),
    .groups    = "drop"
  ) |>
  arrange(desc(MedianCost))

cat("\nCluster cost profile:\n")
cluster_profile |>
  mutate(across(c(MedianCost, MeanCost, P90Cost), ~ scales::dollar(round(., 0)))) |>
  print()

p_cost_cluster <- ggplot(
    df,
    aes(x    = reorder(factor(TextCluster), UltimateIncurredClaimCost, FUN = median),
        y    = log_Target,
        fill = factor(TextCluster))) +
  geom_boxplot(alpha = 0.75, outlier.size = 0.4, outlier.alpha = 0.3) +
  scale_fill_brewer(palette = "Set1", guide = "none") +
  labs(title    = "Claim Cost by Text Cluster",
       subtitle = "Clusters ordered by median cost",
       x = "Text Cluster", y = "log1p(Claim Cost)")
save_plot(p_cost_cluster, "08_cluster_cost_boxplot.png")

# ── 8.6 Sample descriptions per cluster ───────────────────────────────────────
cat("\nSample claim descriptions by cluster:\n")
for (k in sort(unique(df$TextCluster))) {
  examples <- df |>
    filter(TextCluster == k, nchar(ClaimDescription) > 10) |>
    slice_sample(n = 3) |>
    pull(ClaimDescription)
  med_cost <- cluster_profile |> filter(TextCluster == k) |> pull(MedianCost)
  cat(sprintf("Cluster %d [median: %s]:\n", k, scales::dollar(round(med_cost, 0))))
  purrr::walk(examples, ~ cat(sprintf("  → %s\n", .x)))
  cat("\n")
}


# =============================================================================
# SECTION 9 — TRAIN / TEST SPLIT & FEATURE MATRIX
# =============================================================================

cat("\n── Section 9: Train/Test Split & Feature Matrix ──\n")

# ── 9.1 Define feature set ─────────────────────────────────────────────────────
# Explicit list — no identifiers, no raw text, no raw dates, no derived target
FEATURES <- c(
  # Claimant demographics
  "Age",                  # age at accident
  "Gender_enc",           # encoded: M/F/Unknown
  "MaritalStatus_enc",    # encoded: M/S/U/D/Unknown
  "DependentChildren",    # integer count
  "DependentsOther",      # integer count

  # Employment
  "log_WeeklyWages",      # log(pre-injury wages)
  "PartTimeFullTime_enc", # encoded employment type
  "HoursWorkedPerWeek",
  "DaysWorkedPerWeek",

  # Claim financials (early reserve — see leakage note in Section 4)
  "log_InitialCost",      # log(InitialIncurredCalimsCost)

  # Temporal features
  "ReportingDelay",       # IBNR proxy
  "AccidentYear",
  "AccidentMonth",
  "AccidentWeekday",
  "AccidentHour",
  "IsWeekend",

  # NLP feature: BERT → UMAP 10D → K-Means cluster label
  "TextCluster"
)

TARGET <- "log_Target"   # log1p(UltimateIncurredClaimCost)

df_model <- df |>
  select(all_of(c(FEATURES, TARGET))) |>
  drop_na()

cat(sprintf("Model feature matrix: %s rows × %d features\n",
            scales::comma(nrow(df_model)), length(FEATURES)))

# ── 9.2 Stratified 80/20 split ────────────────────────────────────────────────
# Stratify on TextCluster to ensure cluster representation in both sets.
# LEAKAGE RULE: All subsequent stats must be computed on train only.
set.seed(GLOBAL_SEED)
train_idx <- caret::createDataPartition(
  y    = df_model$TextCluster,
  p    = CONFIG$train_frac,
  list = FALSE
)
train_df <- df_model[ train_idx, ]
test_df  <- df_model[-train_idx, ]

cat(sprintf("Train: %s rows | Test: %s rows\n",
            scales::comma(nrow(train_df)),
            scales::comma(nrow(test_df))))

# Verify cluster proportions are similar in train and test
stratification_check <- tibble(
  cluster   = names(table(train_df$TextCluster)),
  train_pct = round(as.numeric(prop.table(table(train_df$TextCluster))) * 100, 1),
  test_pct  = round(as.numeric(prop.table(table(test_df$TextCluster)))  * 100, 1)
)
cat("Cluster proportions (train vs test):\n")
print(stratification_check)

# ── 9.3 XGBoost DMatrix objects ───────────────────────────────────────────────
X_train <- as.matrix(train_df[, FEATURES])
y_train <- train_df[[TARGET]]
X_test  <- as.matrix(test_df[, FEATURES])
y_test  <- test_df[[TARGET]]

dtrain  <- xgboost::xgb.DMatrix(data = X_train, label = y_train)
dtest   <- xgboost::xgb.DMatrix(data = X_test,  label = y_test)
cat("✅ DMatrix objects created.\n")


# =============================================================================
# SECTION 10 — XGBOOST HYPERPARAMETER TUNING
# =============================================================================
# Grid search over key hyperparameters with 5-fold cross-validation.
# Early stopping prevents overfitting by stopping when CV metric stagnates.

cat("\n── Section 10: Hyperparameter Tuning ──\n")

param_grid <- expand.grid(
  max_depth        = c(4L, 6L, 8L),
  eta              = c(0.05, 0.10),
  subsample        = c(0.70, 0.90),
  colsample_bytree = c(0.70, 0.90),
  min_child_weight = c(5L, 10L),
  stringsAsFactors = FALSE
)

cat(sprintf("Grid: %d combinations | %d-fold CV | Early stop: %d rounds\n",
            nrow(param_grid), CONFIG$cv_folds, CONFIG$early_stop))

cv_results <- vector("list", nrow(param_grid))
set.seed(GLOBAL_SEED)

for (i in seq_len(nrow(param_grid))) {
  params_i <- list(
    booster          = "gbtree",
    objective        = "reg:squarederror",
    max_depth        = param_grid$max_depth[i],
    eta              = param_grid$eta[i],
    subsample        = param_grid$subsample[i],
    colsample_bytree = param_grid$colsample_bytree[i],
    min_child_weight = param_grid$min_child_weight[i],
    lambda           = 1.0,       # L2 regularisation
    alpha            = 0.1,       # L1 regularisation
    tree_method      = "hist",    # histogram-based splits (faster)
    eval_metric      = "rmse"
  )

  cv_i <- xgboost::xgb.cv(
    params                = params_i,
    data                  = dtrain,
    nrounds               = CONFIG$max_nrounds,
    nfold                 = CONFIG$cv_folds,
    verbose               = 0,
    early_stopping_rounds = CONFIG$early_stop,
    maximize              = FALSE,
    seed                  = GLOBAL_SEED
  )

  best_round <- which.min(cv_i$evaluation_log$test_rmse_mean)
  best_rmse  <- cv_i$evaluation_log$test_rmse_mean[best_round]
  best_sd    <- cv_i$evaluation_log$test_rmse_std[best_round]

  cv_results[[i]] <- c(
    as.list(param_grid[i, ]),
    list(best_rmse = best_rmse, best_sd = best_sd, best_nround = best_round)
  )

  cat(sprintf("[%3d/%d] d=%d η=%.2f sub=%.1f col=%.1f mcw=%2d | CV-RMSE=%.5f±%.5f  n=%d\n",
              i, nrow(param_grid),
              param_grid$max_depth[i], param_grid$eta[i],
              param_grid$subsample[i], param_grid$colsample_bytree[i],
              param_grid$min_child_weight[i],
              best_rmse, best_sd, best_round))
}

cat("✅ Grid search complete.\n")

# Select best configuration
cv_df    <- bind_rows(lapply(cv_results, as_tibble))
best_row <- cv_df |> slice_min(best_rmse, n = 1)

cat("\nTop 5 configurations:\n")
cv_df |> arrange(best_rmse) |> slice_head(n = 5) |> print()
cat("\nBest configuration:\n")
print(best_row)

best_params <- list(
  booster          = "gbtree",
  objective        = "reg:squarederror",
  max_depth        = best_row$max_depth,
  eta              = best_row$eta,
  subsample        = best_row$subsample,
  colsample_bytree = best_row$colsample_bytree,
  min_child_weight = best_row$min_child_weight,
  lambda           = 1.0,
  alpha            = 0.1,
  tree_method      = "hist",
  eval_metric      = "rmse"
)
best_nrounds <- best_row$best_nround


# =============================================================================
# SECTION 11 — FINAL MODEL TRAINING
# =============================================================================

cat("\n── Section 11: Training Final Model ──\n")

cat(sprintf("Params: depth=%d, η=%.2f, sub=%.1f, col=%.1f, mcw=%d | rounds=%d\n",
            best_params$max_depth, best_params$eta,
            best_params$subsample, best_params$colsample_bytree,
            best_params$min_child_weight, best_nrounds))

set.seed(GLOBAL_SEED)
xgb_final <- xgboost::xgb.train(
  params        = best_params,
  data          = dtrain,
  nrounds       = best_nrounds,
  watchlist     = list(train = dtrain, eval = dtest),
  verbose       = 1,
  print_every_n = max(1L, best_nrounds %/% 10L)
)
cat("✅ Final model trained.\n")

# ── Plot CV learning curve (re-run CV for best params to get full log) ─────────
set.seed(GLOBAL_SEED)
cv_best <- xgboost::xgb.cv(
  params                = best_params,
  data                  = dtrain,
  nrounds               = CONFIG$max_nrounds,
  nfold                 = CONFIG$cv_folds,
  verbose               = 0,
  early_stopping_rounds = CONFIG$early_stop,
  maximize              = FALSE,
  seed                  = GLOBAL_SEED
)

cv_log <- cv_best$evaluation_log

p_learning <- ggplot(cv_log, aes(x = iter)) +
  geom_ribbon(aes(ymin = train_rmse_mean - train_rmse_std,
                  ymax = train_rmse_mean + train_rmse_std),
              fill = "#1565C0", alpha = 0.20) +
  geom_line(aes(y = train_rmse_mean, colour = "Train"), linewidth = 0.8) +
  geom_ribbon(aes(ymin = test_rmse_mean - test_rmse_std,
                  ymax = test_rmse_mean + test_rmse_std),
              fill = "#E53935", alpha = 0.20) +
  geom_line(aes(y = test_rmse_mean, colour = "CV Test"), linewidth = 0.8) +
  geom_vline(xintercept = best_nrounds, linetype = "dashed",
             colour = "grey40", linewidth = 0.7) +
  scale_colour_manual(values = c("Train" = "#1565C0", "CV Test" = "#E53935")) +
  labs(title    = "XGBoost Learning Curve",
       subtitle = paste0("5-fold CV RMSE ± 1SD | Optimal rounds = ", best_nrounds),
       x = "Boosting Round", y = "RMSE (log scale)", colour = NULL)
save_plot(p_learning, "09_learning_curve.png")

# ── Built-in gain-based feature importance ─────────────────────────────────────
importance_matrix <- xgboost::xgb.importance(
  feature_names = FEATURES,
  model         = xgb_final
)
cat("\nXGBoost Feature Importance (Gain):\n")
print(importance_matrix)

png(file.path(CONFIG$output_dir, "10_xgb_importance.png"),
    width = 1000, height = 700, res = 130)
xgboost::xgb.plot.importance(
  importance_matrix = importance_matrix,
  top_n  = length(FEATURES),
  main   = "XGBoost Feature Importance (Gain)"
)
invisible(dev.off())
cat("   Saved: plots/10_xgb_importance.png\n")


# =============================================================================
# SECTION 12 — MODEL EVALUATION
# =============================================================================
# Evaluate on the held-out test set. All predictions are back-transformed
# from log scale via expm1() = exp(x) - 1, the inverse of log1p().

cat("\n── Section 12: Model Evaluation ──\n")

pred_log    <- predict(xgb_final, dtest)
pred_dollar <- expm1(pred_log)
actual_log  <- y_test
actual_dollar <- expm1(actual_log)

compute_metrics <- function(actual, predicted, label = "") {
  rmse <- sqrt(mean((actual - predicted)^2))
  mae  <- mean(abs(actual - predicted))
  r2   <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
  tibble(Scale = label, RMSE = rmse, MAE = mae, R2 = r2)
}

metrics_log    <- compute_metrics(actual_log,    pred_log,    "Log Scale")
metrics_dollar <- compute_metrics(actual_dollar, pred_dollar, "Dollar Scale")

cat("══════════════════════════════════════════\n")
cat("  EVALUATION — HOLD-OUT TEST SET\n")
cat("══════════════════════════════════════════\n")
cat("  Original dollar scale:\n")
cat(sprintf("    RMSE: %s\n", scales::dollar(round(metrics_dollar$RMSE, 0))))
cat(sprintf("    MAE:  %s\n", scales::dollar(round(metrics_dollar$MAE,  0))))
cat(sprintf("    R²:    %.4f\n", metrics_dollar$R2))
cat("  Log scale:\n")
cat(sprintf("    RMSE: %.5f\n", metrics_log$RMSE))
cat(sprintf("    MAE:  %.5f\n", metrics_log$MAE))
cat(sprintf("    R²:    %.4f\n", metrics_log$R2))
cat("══════════════════════════════════════════\n")


# =============================================================================
# SECTION 13 — DIAGNOSTIC PLOTS
# =============================================================================

cat("\n── Section 13: Diagnostic Plots ──\n")

eval_df <- tibble(
  actual_log    = actual_log,
  pred_log      = pred_log,
  actual_dollar = actual_dollar,
  pred_dollar   = pred_dollar,
  residual_log  = actual_log - pred_log,
  residual_pct  = (actual_dollar - pred_dollar) / actual_dollar * 100,
  cluster       = factor(test_df$TextCluster)
)

# 1. Predicted vs Actual
p_pva <- ggplot(eval_df, aes(x = pred_log, y = actual_log, colour = cluster)) +
  geom_point(alpha = 0.25, size = 0.7) +
  geom_abline(slope = 1, intercept = 0, colour = "black",
              linewidth = 1, linetype = "dashed") +
  scale_colour_brewer(palette = "Set1") +
  guides(colour = guide_legend(override.aes = list(size = 3, alpha = 1))) +
  labs(title    = "Predicted vs Actual (Log Scale)",
       subtitle = sprintf("R² = %.4f  |  RMSE = %.4f",
                          metrics_log$R2, metrics_log$RMSE),
       x = "log1p(Predicted)", y = "log1p(Actual)", colour = "Cluster")

# 2. Residuals vs Fitted
p_resid <- ggplot(eval_df, aes(x = pred_log, y = residual_log, colour = cluster)) +
  geom_point(alpha = 0.25, size = 0.7) +
  geom_hline(yintercept = 0, colour = "black", linewidth = 0.9) +
  geom_smooth(method = "loess", se = FALSE, colour = "#E53935",
              linewidth = 0.8, show.legend = FALSE) +
  scale_colour_brewer(palette = "Set1") +
  labs(title    = "Residuals vs Fitted",
       subtitle = "Red LOESS smoother — should be flat at 0",
       x = "log1p(Predicted)", y = "Residual", colour = "Cluster")

# 3. Residual histogram
p_resid_hist <- ggplot(eval_df, aes(x = residual_log)) +
  geom_histogram(bins = 80, fill = "#5C6BC0", alpha = 0.85, colour = "white") +
  geom_vline(xintercept = 0, colour = "#E53935", linewidth = 0.9) +
  geom_vline(xintercept = mean(eval_df$residual_log),
             colour = "#2E7D32", linetype = "dashed", linewidth = 0.8) +
  labs(title    = "Residual Distribution",
       subtitle = sprintf("Mean residual = %.5f (should ≈ 0)",
                          mean(eval_df$residual_log)),
       x = "Residual (log scale)", y = "Count")

# 4. % Error by cluster
p_err_cluster <- ggplot(
    eval_df,
    aes(x    = reorder(cluster, abs(residual_pct), FUN = median),
        y    = abs(residual_pct),
        fill = cluster)) +
  geom_boxplot(alpha = 0.7, outlier.size = 0.4, outlier.alpha = 0.3) +
  scale_fill_brewer(palette = "Set1", guide = "none") +
  scale_y_continuous(limits = c(0, 200),
                     labels = scales::percent_format(scale = 1)) +
  labs(title    = "Absolute % Error by Cluster",
       subtitle = "Higher error clusters may benefit from cluster-specific models",
       x = "Text Cluster", y = "| % Error |")

p_diag_grid <- gridExtra::arrangeGrob(
  p_pva, p_resid, p_resid_hist, p_err_cluster, nrow = 2
)
save_plot(p_diag_grid, "11_model_diagnostics.png", w = 14, h = 10)

# Per-cluster metrics table
cat("\nModel performance by cluster:\n")
eval_df |>
  group_by(cluster) |>
  summarise(
    N        = n(),
    RMSE_log = round(sqrt(mean(residual_log^2)), 4),
    MAE_log  = round(mean(abs(residual_log)), 4),
    R2_log   = round(1 - sum(residual_log^2) /
                     sum((actual_log - mean(actual_log))^2), 4),
    MedActual = scales::dollar(round(median(actual_dollar), 0)),
    .groups  = "drop"
  ) |>
  arrange(desc(RMSE_log)) |>
  print()


# =============================================================================
# SECTION 14 — SHAP EXPLAINABILITY
# =============================================================================
# SHAP (SHapley Additive exPlanations): game-theory-grounded feature attribution.
# Properties: local accuracy, consistency, missingness.
# Uses XGBoost's native TreeSHAP (exact, not approximate).

cat("\n── Section 14: SHAP Explainability ──\n")

library(SHAPforxgboost)
library(shapviz)

# ── 14.1 Compute SHAP values ───────────────────────────────────────────────────
set.seed(GLOBAL_SEED)
shap_n   <- min(CONFIG$shap_n_sample, nrow(X_train))
shap_idx <- sample(nrow(X_train), shap_n)
X_shap   <- X_train[shap_idx, ]

cat(sprintf("Computing SHAP values for %s observations...\n",
            scales::comma(shap_n)))

shap_out  <- SHAPforxgboost::shap.values(xgb_model = xgb_final, X_train = X_shap)
shap_long <- SHAPforxgboost::shap.prep(xgb_model  = xgb_final, X_train = X_shap)

# shapviz objects for waterfall / force / dependence plots
sv_train <- shapviz::shapviz(
  object = xgb_final,
  X_pred = xgboost::xgb.DMatrix(X_shap),
  X      = X_shap
)
sv_test  <- shapviz::shapviz(
  object = xgb_final,
  X_pred = dtest,
  X      = X_test
)
cat("✅ SHAP values computed.\n")

# ── 14.2 Global feature importance (mean |SHAP|) ───────────────────────────────
shap_importance <- shap_out$mean_shap_score |>
  as_tibble(rownames = "feature") |>
  rename(mean_abs_shap = value) |>
  arrange(desc(mean_abs_shap))

cat("\nGlobal SHAP importance:\n")
print(shap_importance)

p_shap_bar <- ggplot(shap_importance,
                     aes(x = reorder(feature, mean_abs_shap),
                         y = mean_abs_shap,
                         fill = mean_abs_shap)) +
  geom_col(width = 0.7) +
  coord_flip() +
  scale_fill_viridis_c(option = "viridis", guide = "none") +
  geom_text(aes(label = round(mean_abs_shap, 4)),
            hjust = -0.1, size = 3.2, colour = "grey30") +
  labs(title    = "Global SHAP Feature Importance",
       subtitle = "Mean |SHAP| across training sample",
       x = NULL, y = "Mean |SHAP| (log scale contribution)")
save_plot(p_shap_bar, "12_shap_importance_bar.png", w = 10, h = 7)

# ── 14.3 SHAP beeswarm summary ─────────────────────────────────────────────────
# Shows direction AND magnitude: x-axis=SHAP, colour=feature value
p_beeswarm <- SHAPforxgboost::shap.plot.summary(shap_long, top_n = length(FEATURES)) +
  labs(title    = "SHAP Summary — Beeswarm",
       subtitle = "Each point = one claim | Red=high feature value, Blue=low",
       caption  = "Positive SHAP → raises predicted cost | Negative → lowers it") +
  theme_actuarial()
save_plot(p_beeswarm, "13_shap_beeswarm.png", w = 10, h = 8)

# shapviz version (alternative style)
p_beeswarm_sv <- shapviz::sv_importance(sv_train, kind = "beeswarm", max_display = 15) +
  labs(title = "SHAP Beeswarm (shapviz)", subtitle = "Top 15 features") +
  theme_actuarial()
save_plot(p_beeswarm_sv, "14_shap_beeswarm_shapviz.png", w = 10, h = 7)

# ── 14.4 SHAP dependence plots (top 4 features) ────────────────────────────────
# Shows how SHAP value changes as feature value changes;
# colour = auto-selected interacting feature
top4_features <- shap_importance$feature[1:4]
cat("Dependence plots for:", paste(top4_features, collapse = ", "), "\n")

dep_plots <- purrr::map(top4_features, function(feat) {
  shapviz::sv_dependence(sv_train, v = feat, color_var = "auto") +
    theme_actuarial(base_size = 10)
})

p_dep_grid <- gridExtra::arrangeGrob(
  grobs = dep_plots, nrow = 2,
  top   = gridExtra::textGrob("SHAP Dependence Plots — Top 4 Features",
                               gp = grid::gpar(fontface = "bold", fontsize = 13))
)
save_plot(p_dep_grid, "15_shap_dependence.png", w = 14, h = 10)

# ── 14.5 InitialCost × TextCluster interaction ─────────────────────────────────
p_interact <- SHAPforxgboost::shap.plot.dependence(
  data_long     = shap_long,
  x             = "log_InitialCost",
  y             = "log_InitialCost",
  color_feature = "TextCluster"
) +
  scale_colour_viridis_c(option = "turbo") +
  labs(title    = "SHAP Dependence: log_InitialCost × TextCluster",
       subtitle = "Colour = TextCluster — reveals cluster-specific cost dynamics",
       x = "log_InitialCost (feature value)", y = "SHAP value") +
  theme_actuarial()
save_plot(p_interact, "16_shap_interaction.png")

# ── 14.6 Individual claim waterfall plots ──────────────────────────────────────
# Explain 3 representative claims: P10 (low), P50 (median), P90 (high cost)
q10_idx <- which.min(abs(actual_dollar - quantile(actual_dollar, 0.10)))
q50_idx <- which.min(abs(actual_dollar - quantile(actual_dollar, 0.50)))
q90_idx <- which.min(abs(actual_dollar - quantile(actual_dollar, 0.90)))

claim_indices <- c(q10_idx, q50_idx, q90_idx)
claim_labels  <- c("Low-Cost (P10)", "Median (P50)", "High-Cost (P90)")
claim_files   <- c("17_shap_waterfall_p10.png",
                   "18_shap_waterfall_p50.png",
                   "19_shap_waterfall_p90.png")

for (j in seq_along(claim_indices)) {
  idx      <- claim_indices[j]
  act_str  <- scales::dollar(round(actual_dollar[idx], 0))
  pred_str <- scales::dollar(round(pred_dollar[idx],  0))

  p_wf <- shapviz::sv_waterfall(sv_test, row_id = idx, max_display = 12) +
    labs(title    = paste0("SHAP Waterfall — ", claim_labels[j]),
         subtitle = sprintf("Actual: %s  |  Predicted: %s", act_str, pred_str)) +
    theme_actuarial(base_size = 11)
  save_plot(p_wf, claim_files[j], w = 10, h = 7)
}

# ── 14.7 Force plot (high-cost claim) ─────────────────────────────────────────
p_force <- shapviz::sv_force(sv_test, row_id = q90_idx) +
  labs(title    = "SHAP Force Plot — High-Cost Claim",
       subtitle = sprintf("Actual: %s  |  Predicted: %s",
                          scales::dollar(round(actual_dollar[q90_idx], 0)),
                          scales::dollar(round(pred_dollar[q90_idx],  0)))) +
  theme_actuarial(base_size = 11)
save_plot(p_force, "20_shap_force_plot.png")

# ── 14.8 TextCluster SHAP contribution analysis ────────────────────────────────
# Does the NLP feature genuinely help? Check its SHAP values per cluster.
shap_cluster_df <- tibble(
  cluster  = factor(X_shap[, "TextCluster"]),
  shap_val = shap_out$shap_score[, "TextCluster"]
)

p_cluster_shap <- ggplot(shap_cluster_df,
                          aes(x = cluster, y = shap_val, fill = cluster)) +
  geom_boxplot(alpha = 0.7, outlier.size = 0.4) +
  geom_hline(yintercept = 0, colour = "black", linewidth = 0.8, linetype = "dashed") +
  scale_fill_brewer(palette = "Set1", guide = "none") +
  labs(title    = "SHAP Value of TextCluster by Group",
       subtitle = "Positive SHAP = cluster increases predicted cost",
       x = "TextCluster", y = "SHAP value",
       caption  = "Validates that NLP clustering captures cost-relevant injury semantics")
save_plot(p_cluster_shap, "21_shap_textcluster.png")


# =============================================================================
# SECTION 15 — MODEL PERSISTENCE & SUMMARY
# =============================================================================

cat("\n── Section 15: Saving Outputs ──\n")

# XGBoost model (JSON: portable, version-stable)
xgboost::xgb.save(xgb_final, CONFIG$model_path)
cat(sprintf("✅ Model saved: %s\n", CONFIG$model_path))

# Test-set predictions with metadata
predictions_out <- test_df |>
  select(TextCluster) |>
  mutate(
    actual_log_cost       = y_test,
    predicted_log_cost    = pred_log,
    actual_dollar_cost    = actual_dollar,
    predicted_dollar_cost = pred_dollar,
    residual_log          = y_test - pred_log,
    pct_error             = (actual_dollar - pred_dollar) / actual_dollar * 100
  )
readr::write_csv(predictions_out, CONFIG$pred_path)
cat(sprintf("✅ Predictions saved: %s\n", CONFIG$pred_path))

# Encoding maps (for future scoring)
saveRDS(encoding_maps, "encoding_maps.rds")
cat("✅ Encoding maps saved: encoding_maps.rds\n")

# Clustering model (K-Means + BERT model name, for new claim scoring)
saveRDS(list(km_model  = km_final,
             optimal_k = optimal_k,
             bert_model= CONFIG$bert_model),
        "clustering_model.rds")
cat("✅ Clustering model saved: clustering_model.rds\n")

# ── Final summary ─────────────────────────────────────────────────────────────
cat("\n")
cat("╔══════════════════════════════════════════════════════════════╗\n")
cat("║     ULTIMATE LOSS PREDICTION — FINAL SUMMARY                ║\n")
cat("╠══════════════════════════════════════════════════════════════╣\n")
cat("║  DATASET                                                      ║\n")
cat(sprintf("║    Claims:         %-12s                           ║\n", scales::comma(nrow(df_raw))))
cat(sprintf("║    Train:          %-12s                           ║\n", scales::comma(nrow(train_df))))
cat(sprintf("║    Test:           %-12s                           ║\n", scales::comma(nrow(test_df))))
cat("╠══════════════════════════════════════════════════════════════╣\n")
cat("║  NLP PIPELINE                                                 ║\n")
cat(sprintf("║    BERT model:     %-38s║\n", CONFIG$bert_model))
cat(sprintf("║    Clusters (K):   %-6d (silhouette = %.4f)        ║\n", optimal_k, best_sil))
cat("╠══════════════════════════════════════════════════════════════╣\n")
cat("║  XGBOOST MODEL                                                ║\n")
cat(sprintf("║    Features:       %-6d                               ║\n", length(FEATURES)))
cat(sprintf("║    Rounds:         %-6d                               ║\n", best_nrounds))
cat(sprintf("║    max_depth:      %-6d                               ║\n", best_params$max_depth))
cat(sprintf("║    eta:            %-6.2f                               ║\n", best_params$eta))
cat("╠══════════════════════════════════════════════════════════════╣\n")
cat("║  PERFORMANCE (TEST SET)                                       ║\n")
cat(sprintf("║    RMSE:           %-28s       ║\n", scales::dollar(round(metrics_dollar$RMSE, 0))))
cat(sprintf("║    MAE:            %-28s       ║\n", scales::dollar(round(metrics_dollar$MAE,  0))))
cat(sprintf("║    R² (dollar):    %-8.4f                             ║\n", metrics_dollar$R2))
cat(sprintf("║    R² (log):       %-8.4f                             ║\n", metrics_log$R2))
cat("╠══════════════════════════════════════════════════════════════╣\n")
cat("║  TOP 3 SHAP FEATURES                                          ║\n")
for (i in 1:3) {
  cat(sprintf("║    %d. %-22s  (mean|SHAP|=%.5f)       ║\n",
              i, shap_importance$feature[i], shap_importance$mean_abs_shap[i]))
}
cat("╠══════════════════════════════════════════════════════════════╣\n")
cat(sprintf("║  Plots saved to: %-44s║\n", paste0(CONFIG$output_dir, "/ (21 files)")))
cat("╚══════════════════════════════════════════════════════════════╝\n")

# ── Session info (reproducibility record) ─────────────────────────────────────
cat("\n── R Session Info ──\n")
sessionInfo()

# =============================================================================
# APPENDIX — Scoring New Claims
# =============================================================================
# To score new claims on a future batch:
#
#   xgb_model     <- xgboost::xgb.load("xgb_model.json")
#   encoding_maps <- readRDS("encoding_maps.rds")
#   cluster_model <- readRDS("clustering_model.rds")
#
#   # 1. Apply same preprocessing (use TRAINING medians, not new data medians)
#   # 2. Generate BERT embeddings on new ClaimDescriptions
#   # 3. Project with saved UMAP model (predict on fitted reducer — if saved)
#   # 4. Assign clusters: predict(cluster_model$km_model, new_umap_10d)$cluster
#   # 5. Score:
#   #     new_dmat  <- xgboost::xgb.DMatrix(as.matrix(new_df[, FEATURES]))
#   #     pred_log  <- predict(xgb_model, new_dmat)
#   #     pred_cost <- expm1(pred_log)   # back to dollars
#
# NOTE: Always use the FITTED clustering/encoding objects — never refit
#       on new data, as that would produce inconsistent cluster labels.
# =============================================================================
