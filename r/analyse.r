################################################################################
# iot_storage_analysis.R
# Comprehensive pipeline: EDA -> modeling (Logistic, Decision Tree, RandomForest)
# Saves plots and model artifacts to folders.
#
# Usage:
# 1) Install required packages (see list below)
# 2) Set data_path to your CSV location
# 3) Run this script
################################################################################

# --------------------------
# Required packages
# --------------------------
packages <- c(
  "readr", "dplyr", "ggplot2", "forcats", "janitor", "corrplot", "GGally",
  "caret", "rpart", "rpart.plot", "randomForest", "pROC", "broom", "tibble"
)
missing <- packages[!(packages %in% installed.packages()[, "Package"])]
if(length(missing)) {
  message("Installing missing packages: ", paste(missing, collapse = ", "))
  install.packages(missing, repos = "https://cloud.r-project.org")
}
invisible(lapply(packages, library, character.only = TRUE))

# --------------------------
# Configuration - update this path if needed
# --------------------------
data_path <- "dataset/IoT_dataset.csv"    # <-- change to your CSV path if different
out_base  <- "analysis_outputs"       # base folder to save everything
set.seed(42)

# --------------------------
# Create output folders
# --------------------------
dirs <- list(
  paste0(out_base, "/data"),
  paste0(out_base, "/plots/eda"),
  paste0(out_base, "/plots/models"),
  paste0(out_base, "/models"),
  paste0(out_base, "/summaries")
)
invisible(lapply(dirs, function(d) if(!dir.exists(d)) dir.create(d, recursive = TRUE)))

# --------------------------
# Read dataset (defensive)
# --------------------------
message("Reading data: ", data_path)
df_raw <- readr::read_csv(data_path, show_col_types = FALSE)

# Save a copy of the raw read for reproducibility
readr::write_csv(df_raw, file.path(out_base, "data", "raw_copy.csv"))

# --------------------------
# Clean column names and auto-detect target
# --------------------------
df <- df_raw %>% janitor::clean_names() %>% as_tibble()
colnames(df) <- make.names(colnames(df), unique = TRUE) # make safe names
message("Clean column names: ", paste(colnames(df), collapse = ", "))

# Auto-detect target column by name containing "output" (case-insensitive)
possible_targets <- grep("output|target|label|accept", names(df), ignore.case = TRUE, value = TRUE)
if(length(possible_targets) == 0) {
  stop("No target column detected. Ensure your target column name contains 'output' or change the auto-detection.")
}
target_col <- possible_targets[1]
message("Chosen target column: ", target_col)

# --------------------------
# Quick diagnostics
# --------------------------
cat("\n---- Data snapshot ----\n")
print(dplyr::glimpse(df))

# --------------------------
# Preprocessing
# --------------------------
# 1) Drop obvious ID-like columns (contains "id")
id_cols <- grep("(^|\\_)id$|_id", names(df), ignore.case = TRUE, value = TRUE)
if(length(id_cols) > 0) {
  message("Dropping ID columns: ", paste(id_cols, collapse = ", "))
  df <- df %>% select(-all_of(id_cols))
}

# 2) Convert the target to factor (nominal) if it's numeric 0/1
if(is.numeric(df[[target_col]])) {
  # If numeric but only has values 0/1 (or close), convert to factor
  vals <- sort(unique(na.omit(df[[target_col]])))
  if(all(vals %in% c(0,1))) {
    df[[target_col]] <- factor(df[[target_col]], levels = c(0,1), labels = c("no","yes"))
    message("Converted numeric 0/1 target to factor with levels: no, yes")
  } else {
    # for other numeric targets, binarize at median as fallback (but warn)
    med <- median(df[[target_col]], na.rm = TRUE)
    df[[target_col]] <- factor(ifelse(df[[target_col]] <= med, "no", "yes"))
    message("Binarized numeric target at median (fallback). Please check if this is desired.")
  }
} else {
  # ensure factor
  df[[target_col]] <- as.factor(df[[target_col]])
}

# 3) Detect categorical columns and collapse very high cardinality
cat_cols <- names(df)[sapply(df, function(x) is.character(x) || is.factor(x))]
cat_cols <- setdiff(cat_cols, target_col)
message("Categorical columns detected: ", paste(cat_cols, collapse = ", "))

# Convert character -> factor and lump levels > threshold
max_levels <- 10   # collapse to top 10 levels + "Other"
for(col in cat_cols) {
  df[[col]] <- as.factor(df[[col]])
  if(nlevels(df[[col]]) > max_levels) {
    df[[col]] <- forcats::fct_lump_n(df[[col]], n = max_levels, other_level = "Other")
    message(sprintf("Collapsed %s to top %d levels (+Other). New levels: %d", col, max_levels, nlevels(df[[col]])))
  }
}

# 4) Ensure numeric columns are numeric
num_cols <- names(df)[sapply(df, is.numeric)]
message("Numeric columns: ", paste(num_cols, collapse = ", "))

# 5) Summarize missingness
missing_summary <- sapply(df, function(x) sum(is.na(x)))
missing_summary <- tibble::enframe(missing_summary, name = "variable", value = "n_missing")
readr::write_csv(missing_summary, file.path(out_base, "summaries", "missing_summary.csv"))
message("Missingness summary saved.")

# Optional simple strategy: drop rows with missing target, impute numeric with median
df <- df %>% filter(!is.na(.data[[target_col]]))
for(nm in num_cols) {
  if(any(is.na(df[[nm]]))) {
    med <- median(df[[nm]], na.rm = TRUE)
    df[[nm]][is.na(df[[nm]])] <- med
    message(sprintf("Imputed median for %s", nm))
  }
}

# --------------------------
# EDA: distributions, boxplots, correlation
# --------------------------
# Helper: safe ggsave wrapper
ggsave_safe <- function(plot, filename, width=8, height=6, dpi=300) {
  ggsave(filename = filename, plot = plot, width = width, height = height, dpi = dpi)
}

# 1) Numeric histograms
num_cols <- names(df)[sapply(df, is.numeric)]
for(col in num_cols) {
  p <- ggplot(df, aes_string(x = col)) +
    geom_histogram(bins = 40, color = "black", fill = "grey80") +
    ggtitle(paste("Distribution:", col)) +
    theme_minimal()
  fname <- file.path(out_base, "plots", "eda", paste0("hist_", col, ".png"))
  ggsave_safe(p, fname)
}

# 2) Boxplots numeric vs target
for(col in num_cols) {
  p <- ggplot(df, aes_string(x = target_col, y = col)) +
    geom_boxplot() +
    ggtitle(paste("Boxplot:", col, "by", target_col)) +
    theme_minimal()
  fname <- file.path(out_base, "plots", "eda", paste0("box_", col, "_by_", target_col, ".png"))
  ggsave_safe(p, fname)
}

# 3) Pairwise scatter (GGally) for the top numeric features (limit to 8)
pair_cols <- head(num_cols, 8)
if(length(pair_cols) >= 2) {
  p <- GGally::ggpairs(df %>% dplyr::select(all_of(pair_cols)))
  fname <- file.path(out_base, "plots", "eda", paste0("pairs_", paste(pair_cols, collapse="_"), ".png"))
  # ggsave for ggpairs requires a slightly different approach
  png(fname, width = 2000, height = 2000, res = 200)
  print(p)
  dev.off()
}

# 4) Correlation heatmap (numerics only)
if(length(num_cols) >= 2) {
  cm <- cor(df %>% dplyr::select(all_of(num_cols)), use = "pairwise.complete.obs")
  png(file.path(out_base, "plots", "eda", "correlation_heatmap.png"), width=1200, height=900, res=150)
  corrplot::corrplot(cm, method = "color", tl.cex = 0.8, number.cex = 0.7)
  dev.off()
}

# 5) Save distributions summary
desc_summary <- df %>% dplyr::select(all_of(num_cols)) %>% summary()
capture.output(desc_summary, file = file.path(out_base, "summaries", "numeric_summary.txt"))

# --------------------------
# Modeling prep: train/test split via caret (we'll use 10-fold CV)
# --------------------------
df_model <- df
# caret training control
train_ctrl <- caret::trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  verboseIter = FALSE
)

# Make sure the target has positive class "yes" and negative "no"
# If not, relevel
if(!all(levels(df_model[[target_col]]) %in% c("no","yes"))) {
  # map first unique to no/yes
  lv <- levels(df_model[[target_col]])
  lv_map <- setNames(c("no","yes")[1:length(lv)], lv)
  df_model[[target_col]] <- factor(as.character(df_model[[target_col]]), levels = names(lv_map))
  levels(df_model[[target_col]]) <- c("no","yes")[1:length(lv)]
}
df_model[[target_col]] <- relevel(df_model[[target_col]], ref = "no")

# Partition: (caret will do CV, but we create a hold-out test optionally)
set.seed(42)
train_index <- caret::createDataPartition(df_model[[target_col]], p = 0.8, list = FALSE)
train_df <- df_model[train_index, ]
test_df  <- df_model[-train_index, ]

# Save splitted data
readr::write_csv(train_df, file.path(out_base, "data", "train.csv"))
readr::write_csv(test_df, file.path(out_base, "data", "test.csv"))

# --------------------------
# Model 1: Logistic Regression (glm)
# --------------------------
message("Training logistic regression (glm) ...")
glm_fit <- caret::train(
  form = as.formula(paste(target_col, "~ .")),
  data = train_df,
  method = "glm",
  family = binomial(),
  trControl = train_ctrl,
  metric = "ROC"
)
saveRDS(glm_fit, file.path(out_base, "models", "glm_fit.rds"))
capture.output(summary(glm_fit), file = file.path(out_base, "summaries", "glm_summary.txt"))
message("GLM saved.")

# Evaluate on test set
glm_pred_prob <- predict(glm_fit, test_df, type = "prob")[, "yes"]
glm_pred_class <- predict(glm_fit, test_df)
glm_cm <- caret::confusionMatrix(glm_pred_class, test_df[[target_col]], positive = "yes")
capture.output(glm_cm, file = file.path(out_base, "summaries", "glm_confusion_matrix.txt"))

# ROC plot
glm_roc <- pROC::roc(response = test_df[[target_col]], predictor = glm_pred_prob, levels = c("no","yes"))
png(file.path(out_base, "plots", "models", "glm_roc.png"), width = 1000, height = 800, res = 150)
plot(glm_roc, main = paste("GLM ROC AUC:", round(pROC::auc(glm_roc), 4)))
dev.off()

# --------------------------
# Model 2: Decision Tree (rpart)
# --------------------------
message("Training decision tree (rpart) ...")
rpart_fit <- caret::train(
  form = as.formula(paste(target_col, "~ .")),
  data = train_df,
  method = "rpart",
  trControl = train_ctrl,
  tuneLength = 10,
  metric = "ROC"
)
saveRDS(rpart_fit, file.path(out_base, "models", "rpart_fit.rds"))
capture.output(rpart_fit, file = file.path(out_base, "summaries", "rpart_summary.txt"))

# plot tree
png(file.path(out_base, "plots", "models", "rpart_tree.png"), width = 1200, height = 900, res = 150)
rpart.plot::rpart.plot(rpart::rpart(as.formula(paste(target_col, "~ .")), data = train_df))
dev.off()

# Evaluate tree on test set
rpart_pred_prob <- predict(rpart_fit, test_df, type = "prob")[, "yes"]
rpart_pred_class <- predict(rpart_fit, test_df)
rpart_cm <- caret::confusionMatrix(rpart_pred_class, test_df[[target_col]], positive = "yes")
capture.output(rpart_cm, file = file.path(out_base, "summaries", "rpart_confusion_matrix.txt"))

rpart_roc <- pROC::roc(response = test_df[[target_col]], predictor = rpart_pred_prob, levels = c("no","yes"))
png(file.path(out_base, "plots", "models", "rpart_roc.png"), width = 1000, height = 800, res = 150)
plot(rpart_roc, main = paste("RPart ROC AUC:", round(pROC::auc(rpart_roc), 4)))
dev.off()

# --------------------------
# Model 3: Random Forest
# --------------------------
message("Training Random Forest ...")
# randomForest can be slow for very large datasets; caret wraps it with tuning
rf_fit <- caret::train(
  form = as.formula(paste(target_col, "~ .")),
  data = train_df,
  method = "rf",
  trControl = train_ctrl,
  tuneLength = 6,
  metric = "ROC",
  ntree = 200
)
saveRDS(rf_fit, file.path(out_base, "models", "rf_fit.rds"))
capture.output(rf_fit, file = file.path(out_base, "summaries", "rf_summary.txt"))

# Variable importance
varimp_rf <- varImp(rf_fit, scale = TRUE)
png(file.path(out_base, "plots", "models", "rf_varimp.png"), width = 1200, height = 900, res = 150)
plot(varimp_rf, top = 30, main = "RandomForest Variable Importance")
dev.off()
capture.output(varimp_rf, file = file.path(out_base, "summaries", "rf_varimp.txt"))

# Evaluate on test
rf_pred_prob <- predict(rf_fit, test_df, type = "prob")[, "yes"]
rf_pred_class <- predict(rf_fit, test_df)
rf_cm <- caret::confusionMatrix(rf_pred_class, test_df[[target_col]], positive = "yes")
capture.output(rf_cm, file = file.path(out_base, "summaries", "rf_confusion_matrix.txt"))

rf_roc <- pROC::roc(response = test_df[[target_col]], predictor = rf_pred_prob, levels = c("no","yes"))
png(file.path(out_base, "plots", "models", "rf_roc.png"), width = 1000, height = 800, res = 150)
plot(rf_roc, main = paste("RF ROC AUC:", round(pROC::auc(rf_roc), 4)))
dev.off()

# --------------------------
# Compare models on test set - write summary
# --------------------------
model_results <- tibble::tibble(
  model = c("GLM", "RPART", "RF"),
  accuracy = c(glm_cm$overall["Accuracy"], rpart_cm$overall["Accuracy"], rf_cm$overall["Accuracy"]),
  kappa = c(glm_cm$overall["Kappa"], rpart_cm$overall["Kappa"], rf_cm$overall["Kappa"]),
  sensitivity = c(glm_cm$byClass["Sensitivity"], rpart_cm$byClass["Sensitivity"], rf_cm$byClass["Sensitivity"]),
  specificity = c(glm_cm$byClass["Specificity"], rpart_cm$byClass["Specificity"], rf_cm$byClass["Specificity"]),
  auc = c(as.numeric(pROC::auc(glm_roc)), as.numeric(pROC::auc(rpart_roc)), as.numeric(pROC::auc(rf_roc)))
)
readr::write_csv(model_results, file.path(out_base, "summaries", "model_comparison.csv"))
capture.output(model_results, file = file.path(out_base, "summaries", "model_comparison.txt"))
message("Model comparison saved.")

# --------------------------
# Save final predictions & important outputs
# --------------------------
predictions_df <- test_df %>% 
  dplyr::select(-all_of(target_col)) %>%
  mutate(
    true = test_df[[target_col]],
    glm_prob = glm_pred_prob, glm_pred = glm_pred_class,
    rpart_prob = rpart_pred_prob, rpart_pred = rpart_pred_class,
    rf_prob = rf_pred_prob, rf_pred = rf_pred_class
  )
readr::write_csv(predictions_df, file.path(out_base, "data", "test_predictions.csv"))

# --------------------------
# Save session info and script snapshot
# --------------------------
capture.output(sessionInfo(), file = file.path(out_base, "summaries", "session_info.txt"))
# Also save a copy of this script if running interactively
script_path <- "iot_storage_analysis.R"
if(file.exists(script_path)) {
  file.copy(script_path, file.path(out_base, "data", basename(script_path)), overwrite = TRUE)
}

message("All done. Outputs are under: ", out_base)
