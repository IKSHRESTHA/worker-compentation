# cluster_claims.R
# Text cluster analysis on ClaimDescription using TF-IDF + K-means (LSA).
#
# Workflow:
#   1.  Load & filter data
#   2.  Tokenise, remove stopwords, clean text
#   3.  Build TF-IDF document-term matrix
#   4.  Reduce dimensions with truncated SVD (Latent Semantic Analysis)
#   5.  Choose optimal k via elbow + silhouette
#   6.  Fit final K-means model
#   7.  Profile clusters (top TF-IDF terms)
#   8.  Visualise (PCA scatter, top-terms bar chart)
#   9.  Attach cluster labels to original data & export
#
# Required packages (install once with renv::install() or install.packages()):
#   tidytext, cluster, ggplot2, dplyr, tidyr, stringr, readr, here


library(here)
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(tidytext)   # tokenisation, stop_words, bind_tf_idf, reorder_within
library(cluster)    # silhouette() — ships with base R

set.seed(42)        # reproducibility

# ── 1. Load data ──────────────────────────────────────────────────────────────
DATA_PATH <- here("data", "raw", "data.csv")

col_spec <- cols(
  ClaimNumber               = col_character(),
  DateTimeOfAccident        = col_datetime(format = "%Y-%m-%dT%H:%M:%SZ"),
  DateReported              = col_datetime(format = "%Y-%m-%dT%H:%M:%SZ"),
  Age                       = col_integer(),
  Gender                    = col_factor(levels = c("M", "F")),
  MaritalStatus             = col_factor(levels = c("M", "S", "U", "D", "W")),
  DependentChildren         = col_integer(),
  DependentsOther           = col_integer(),
  WeeklyWages               = col_double(),
  PartTimeFullTime          = col_factor(levels = c("F", "P")),
  HoursWorkedPerWeek        = col_double(),
  DaysWorkedPerWeek         = col_double(),
  ClaimDescription          = col_character(),
  InitialIncurredCalimsCost = col_double(),
  UltimateIncurredClaimCost = col_double()
)

claims <- read_csv(DATA_PATH, col_types = col_spec)

# Keep only claims with a usable description; assign a stable doc_id
claims_text <- claims |>
  filter(!is.na(ClaimDescription), str_trim(ClaimDescription) != "") |>
  mutate(doc_id = row_number())

cat(sprintf("Claims with descriptions: %d\n", nrow(claims_text)))

# ── 2. Text preprocessing ─────────────────────────────────────────────────────
# Lower-case → strip non-alpha → tokenise → drop stopwords → drop short tokens
tokens <- claims_text |>
  select(doc_id, ClaimDescription) |>
  mutate(
    text = str_to_lower(ClaimDescription),
    text = str_remove_all(text, "[^a-z\\s]"),
    text = str_squish(text)
  ) |>
  unnest_tokens(word, text) |>
  anti_join(stop_words, by = "word") |>
  filter(str_length(word) > 2)

# ── 3. TF-IDF document-term matrix ───────────────────────────────────────────
tfidf <- tokens |>
  count(doc_id, word) |>
  bind_tf_idf(word, doc_id, n)

# Restrict vocabulary: keep the top 500 terms by corpus-wide TF-IDF weight.
# Smaller feature spaces cluster faster and reduce noise.
top_terms <- tfidf |>
  group_by(word) |>
  summarise(total_tfidf = sum(tf_idf), .groups = "drop") |>
  slice_max(total_tfidf, n = 500) |>
  pull(word)

dtm_wide <- tfidf |>
  filter(word %in% top_terms) |>
  select(doc_id, word, tf_idf) |>
  pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0) |>
  arrange(doc_id)

doc_ids <- dtm_wide$doc_id
dtm_mat <- as.matrix(dtm_wide[, -1])          # numeric matrix (docs × terms)

# ── 4. Dimensionality reduction: truncated SVD (LSA) ─────────────────────────
# LSA removes noise and collinearity that hurt distance-based clustering.
# Keep enough singular components to explain ≥ 80 % of corpus variance.
n_sv       <- min(50L, ncol(dtm_mat) - 1L, nrow(dtm_mat) - 1L)
svd_result <- svd(dtm_mat, nu = n_sv, nv = n_sv)

var_explained <- cumsum(svd_result$d[seq_len(n_sv)]^2) /
                   sum(svd_result$d^2)

n_components <- which(var_explained >= 0.80)[1]
n_components <- ifelse(is.na(n_components), n_sv, n_components)

# Projected document coordinates in LSA space
lsa_scores <- svd_result$u[, seq_len(n_components), drop = FALSE] %*%
              diag(svd_result$d[seq_len(n_components)], nrow = n_components)

cat(sprintf("LSA: retaining %d components (%.1f %% variance)\n",
            n_components, 100 * var_explained[n_components]))

# ── 5. Choose optimal k ───────────────────────────────────────────────────────
k_range <- 2:10

# -- 5a. Elbow (total within-cluster SSE) -------------------------------------
wss <- vapply(k_range, function(k) {
  kmeans(lsa_scores, centers = k, nstart = 25, iter.max = 100)$tot.withinss
}, numeric(1))

p_elbow <- ggplot(tibble(k = k_range, wss = wss), aes(k, wss)) +
  geom_line(colour = "#2C7BB6", linewidth = 0.9) +
  geom_point(size = 3, colour = "#2C7BB6") +
  scale_x_continuous(breaks = k_range) +
  labs(
    title    = "Elbow Method – Within-Cluster SSE",
    subtitle = "Look for the 'elbow' where the drop flattens",
    x        = "Number of clusters (k)",
    y        = "Total within-cluster SSE"
  ) +
  theme_minimal(base_size = 13)

ggsave(here("reports", "figures", "cluster_elbow.png"),
       p_elbow, width = 7, height = 4, dpi = 150)

# -- 5b. Average silhouette width (higher = better-defined clusters) ----------
sil_scores <- vapply(k_range, function(k) {
  km  <- kmeans(lsa_scores, centers = k, nstart = 25, iter.max = 100)
  sil <- silhouette(km$cluster, dist(lsa_scores))
  mean(sil[, "sil_width"])
}, numeric(1))

sil_df <- tibble(k = k_range, silhouette = sil_scores)
best_k  <- sil_df$k[which.max(sil_df$silhouette)]

cat(sprintf("Optimal k by silhouette: %d  (avg sil = %.3f)\n",
            best_k, max(sil_scores)))

p_sil <- ggplot(sil_df, aes(k, silhouette)) +
  geom_line(colour = "#D7191C", linewidth = 0.9) +
  geom_point(size = 3, colour = "#D7191C") +
  geom_vline(xintercept = best_k, linetype = "dashed", colour = "grey40") +
  annotate("text", x = best_k + 0.15, y = min(sil_scores),
           label = paste("Best k =", best_k), hjust = 0, size = 3.5) +
  scale_x_continuous(breaks = k_range) +
  labs(
    title    = "Silhouette Analysis",
    subtitle = "Higher average silhouette = more cohesive, well-separated clusters",
    x        = "Number of clusters (k)",
    y        = "Average silhouette width"
  ) +
  theme_minimal(base_size = 13)

ggsave(here("reports", "figures", "cluster_silhouette.png"),
       p_sil, width = 7, height = 4, dpi = 150)

# ── 6. Fit final K-means model ────────────────────────────────────────────────
# nstart = 50: run 50 random initialisations and keep the best result.
km_final <- kmeans(lsa_scores, centers = best_k, nstart = 50, iter.max = 200)

claims_text <- claims_text |>
  mutate(cluster = factor(km_final$cluster))

cat("\nCluster sizes:\n")
print(table(claims_text$cluster))

# ── 7. Profile clusters: top TF-IDF terms per cluster ────────────────────────
cluster_terms <- tokens |>
  inner_join(claims_text |> select(doc_id, cluster), by = "doc_id") |>
  count(cluster, word) |>
  bind_tf_idf(word, cluster, n) |>
  group_by(cluster) |>
  slice_max(tf_idf, n = 10, with_ties = FALSE) |>
  ungroup()

p_terms <- ggplot(
    cluster_terms,
    aes(tf_idf, reorder_within(word, tf_idf, cluster), fill = cluster)
  ) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~cluster, scales = "free_y") +
  scale_y_reordered() +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title    = "Top 10 Discriminating Terms per Cluster",
    subtitle = "Ranked by within-cluster TF-IDF",
    x        = "TF-IDF",
    y        = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(strip.text = element_text(face = "bold"))

ggsave(here("reports", "figures", "cluster_top_terms.png"),
       p_terms, width = 5 * ceiling(sqrt(best_k)), height = 4 * ceiling(best_k / ceiling(sqrt(best_k))),
       dpi = 150, limitsize = FALSE)

# ── 8. 2-D visualisation via PCA on LSA scores ───────────────────────────────
pca_2d  <- prcomp(lsa_scores, center = TRUE, scale. = FALSE)
pve     <- round(100 * pca_2d$sdev[1:2]^2 / sum(pca_2d$sdev^2), 1)

pca_df  <- as_tibble(pca_2d$x[, 1:2]) |>
  mutate(cluster = claims_text$cluster)

p_pca <- ggplot(pca_df, aes(PC1, PC2, colour = cluster)) +
  geom_point(alpha = 0.45, size = 1.6) +
  scale_colour_brewer(palette = "Set2") +
  labs(
    title    = "K-means Clusters – PCA Projection of LSA Space",
    subtitle = sprintf("k = %d  |  PC1 explains %.1f %%, PC2 explains %.1f %% of LSA variance",
                       best_k, pve[1], pve[2]),
    x        = sprintf("PC1 (%.1f %%)", pve[1]),
    y        = sprintf("PC2 (%.1f %%)", pve[2]),
    colour   = "Cluster"
  ) +
  theme_minimal(base_size = 13)

ggsave(here("reports", "figures", "cluster_pca.png"),
       p_pca, width = 7, height = 5, dpi = 150)

# ── 9. Numeric summary per cluster ───────────────────────────────────────────
cluster_summary <- claims |>
  mutate(doc_id = row_number()) |>
  inner_join(claims_text |> select(doc_id, cluster), by = "doc_id") |>
  group_by(cluster) |>
  summarise(
    n                       = n(),
    avg_age                 = round(mean(Age,                    na.rm = TRUE), 1),
    avg_weekly_wages        = round(mean(WeeklyWages,            na.rm = TRUE), 2),
    avg_ultimate_claim_cost = round(mean(UltimateIncurredClaimCost, na.rm = TRUE), 2),
    pct_full_time           = round(100 * mean(PartTimeFullTime == "F", na.rm = TRUE), 1),
    .groups = "drop"
  )

cat("\nCluster summary (numeric features):\n")
print(cluster_summary)

# ── 10. Export labelled claims ────────────────────────────────────────────────
write_csv(
  claims_text |> select(doc_id, ClaimNumber, ClaimDescription, cluster),
  here("data", "processed", "claims_clustered.csv")
)

cat("\nDone.\n",
    "  Plots   → reports/figures/cluster_elbow.png\n",
    "           reports/figures/cluster_silhouette.png\n",
    "           reports/figures/cluster_top_terms.png\n",
    "           reports/figures/cluster_pca.png\n",
    "  Data    → data/processed/claims_clustered.csv\n")

