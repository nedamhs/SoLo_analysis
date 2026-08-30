library(dplyr)
library(purrr)
library(readr)
library(tidyr)
library(corrplot)

DATASET_DIR <- "/Users/nedamohseni/Downloads/SoLo_dataset"

# load daily EMA data across participants
participant_dirs <- list.dirs(DATASET_DIR, recursive = FALSE)

ema_df <- map_dfr(participant_dirs, function(participant_dir) {
  
  ema_path <- file.path(participant_dir, "Self_Report", "ema_daily.csv")
  
  if (!file.exists(ema_path)) return(NULL)
  
  read_csv(ema_path, show_col_types = FALSE) |>
    mutate(participant = basename(participant_dir))
})

cat("Loaded participants:", n_distinct(ema_df$participant), "\n")
cat("Total EMA rows:", format(nrow(ema_df), big.mark = ","), "\n")


EMA_VARS <- c("affect_positive", "affect_negative",
              "feel_lonely", "feel_isolated", "feel_connected",
              "number_social_interactions", "pleasant_social_interactions")


# calculate within-participant pairwise Pearson correlations
participant_correlations <- ema_df |>
  group_by(participant) |>
  group_modify(~ {
    cor_matrix <- cor(.x[EMA_VARS], use = "pairwise.complete.obs", method = "pearson")
    
    idx <- which(upper.tri(cor_matrix), arr.ind = TRUE)
    
    tibble(
      var1 = rownames(cor_matrix)[idx[, 1]],
      var2 = colnames(cor_matrix)[idx[, 2]],
      r = cor_matrix[idx]
    )
  }) |>
  ungroup()


# aggregate correlations using equal-weight Fisher-z
correlation_summary <- participant_correlations |>
  mutate(
    r = pmin(pmax(r, -0.999999), 0.999999),
    fisher_z = atanh(r)
  ) |>
  group_by(var1, var2) |>
  summarise(
    mean_z = mean(fisher_z, na.rm = TRUE),
    r = tanh(mean_z),
    n_participants = sum(!is.na(fisher_z)),
    .groups = "drop"
  )


# calculate p-values for participant-level Fisher-z correlations
correlation_summary <- correlation_summary |>
  left_join(
    participant_correlations |>
      mutate(
        r = pmin(pmax(r, -0.999999), 0.999999),
        fisher_z = atanh(r)
      ) |>
      group_by(var1, var2) |>
      summarise(
        p = t.test(fisher_z, mu = 0)$p.value,
        .groups = "drop"
      ),
    by = c("var1", "var2")
  ) |>
  mutate(
    p_stars = case_when(
      p < 0.001 ~ "***",
      p < 0.01  ~ "**",
      p < 0.05  ~ "*",
      TRUE      ~ ""
    )
  )

# adjust p-values for multiple comparisons using Benjamini-Hochberg FDR
correlation_summary <- correlation_summary |>
  mutate(
    p_FDR = p.adjust(p, method = "BH"),
    p_FDR_stars = case_when(
      p_FDR < 0.001 ~ "***",
      p_FDR < 0.01  ~ "**",
      p_FDR < 0.05  ~ "*",
      TRUE          ~ ""
    )
  )

# correlation matrix
cor_matrix <- diag(length(EMA_VARS))
rownames(cor_matrix) <- EMA_VARS
colnames(cor_matrix) <- EMA_VARS

for (i in seq_len(nrow(correlation_summary))) {
  v1 <- correlation_summary$var1[i]
  v2 <- correlation_summary$var2[i]
  r  <- correlation_summary$r[i]
  
  cor_matrix[v1, v2] <- r
  cor_matrix[v2, v1] <- r
}

# rename variables
display_labels <- c(
  "affect_positive" = "Positive Affect",
  "affect_negative" = "Negative Affect",
  "feel_lonely" = "Feeling Lonely",
  "feel_isolated" = "Feeling Isolated",
  "feel_connected" = "Feeling Connected",
  "number_social_interactions" = "Number of Social Interactions",
  "pleasant_social_interactions" = "Recent Interaction Pleasantness"
)

rownames(cor_matrix) <- display_labels[rownames(cor_matrix)]
colnames(cor_matrix) <- display_labels[colnames(cor_matrix)]

corrplot(
  cor_matrix,
  method = "circle",
  type = "lower",
  diag = FALSE,
  order = "original",
  tl.col = "black",
  tl.srt = 45,
  tl.cex = 0.9
)


