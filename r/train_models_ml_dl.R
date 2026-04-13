################################################################################
# train_models_ml_dl.R
# Train multiple ML models (R-native), compare them, and select the best model.
#
# Usage:
#   Rscript r/train_models_ml_dl.R
#
# Outputs:
#   r/ml_dl_outputs/
#     - summaries/model_comparison.csv
#     - summaries/best_model_summary.txt
#     - models/best_model.rds
################################################################################

required_packages <- c(
  "readr", "dplyr", "janitor", "caret", "pROC", "rpart", "randomForest", "nnet", "tibble", "glmnet", "xgboost"
)

missing_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(missing_packages) > 0) {
  message("Installing missing packages: ", paste(missing_packages, collapse = ", "))
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}

invisible(lapply(required_packages, function(pkg) {
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}))

set.seed(42)

# --------------------------
# Configuration
# --------------------------
data_candidates <- c("dataset/IoT_dataset_cleaned.csv", "dataset/IoT_dataset.csv")
data_path <- data_candidates[file.exists(data_candidates)][1]
if (is.na(data_path) || length(data_path) == 0) {
  stop("No dataset found. Expected one of: dataset/IoT_dataset_cleaned.csv or dataset/IoT_dataset.csv")
}

out_base <- "r/ml_dl_outputs"
dirs <- c(
  file.path(out_base, "data"),
  file.path(out_base, "models"),
  file.path(out_base, "summaries")
)
invisible(lapply(dirs, function(d) if (!dir.exists(d)) dir.create(d, recursive = TRUE)))

message("Reading dataset: ", data_path)
df_raw <- readr::read_csv(data_path, show_col_types = FALSE)
readr::write_csv(df_raw, file.path(out_base, "data", "raw_copy.csv"))

# --------------------------
# Preprocessing
# --------------------------
df <- df_raw %>% janitor::clean_names() %>% as_tibble()
colnames(df) <- make.names(colnames(df), unique = TRUE)

target_candidates <- grep("output|target|label|accept", names(df), ignore.case = TRUE, value = TRUE)
if (length(target_candidates) == 0) {
  stop("Target column not detected. Ensure target name contains output/target/label/accept.")
}
target_col <- target_candidates[1]
message("Target column: ", target_col)

id_cols <- grep("(^|\\_)id$|_id", names(df), ignore.case = TRUE, value = TRUE)
if (length(id_cols) > 0) {
  message("Dropping ID columns: ", paste(id_cols, collapse = ", "))
  df <- df %>% dplyr::select(-all_of(id_cols))
}

if (is.numeric(df[[target_col]])) {
  vals <- sort(unique(na.omit(df[[target_col]])))
  if (all(vals %in% c(0, 1))) {
    df[[target_col]] <- factor(df[[target_col]], levels = c(0, 1), labels = c("no", "yes"))
  } else {
    med <- median(df[[target_col]], na.rm = TRUE)
    df[[target_col]] <- factor(ifelse(df[[target_col]] <= med, "no", "yes"), levels = c("no", "yes"))
  }
} else {
  df[[target_col]] <- as.factor(df[[target_col]])
  if (nlevels(df[[target_col]]) != 2) {
    stop("This script currently supports binary classification only.")
  }
  levels(df[[target_col]]) <- c("no", "yes")
}

df <- df %>% dplyr::filter(!is.na(.data[[target_col]]))

for (nm in names(df)) {
  if (nm == target_col) next
  if (is.numeric(df[[nm]]) && any(is.na(df[[nm]]))) {
    med <- median(df[[nm]], na.rm = TRUE)
    df[[nm]][is.na(df[[nm]])] <- med
  }
  if ((is.character(df[[nm]]) || is.factor(df[[nm]])) && any(is.na(df[[nm]]))) {
    df[[nm]][is.na(df[[nm]])] <- "missing"
    df[[nm]] <- as.factor(df[[nm]])
  }
}

set.seed(42)
train_index <- caret::createDataPartition(df[[target_col]], p = 0.8, list = FALSE)
train_df <- df[train_index, ]
test_df <- df[-train_index, ]

readr::write_csv(train_df, file.path(out_base, "data", "train.csv"))
readr::write_csv(test_df, file.path(out_base, "data", "test.csv"))

train_ctrl <- caret::trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

form <- as.formula(paste(target_col, "~ ."))

evaluate_binary <- function(y_true, y_prob, threshold = 0.5) {
  y_pred <- factor(ifelse(y_prob >= threshold, "yes", "no"), levels = c("no", "yes"))
  cm <- caret::confusionMatrix(y_pred, y_true, positive = "yes")
  auc_val <- as.numeric(pROC::auc(pROC::roc(response = y_true, predictor = y_prob, levels = c("no", "yes"), quiet = TRUE)))
  precision <- as.numeric(cm$byClass["Pos Pred Value"])
  recall <- as.numeric(cm$byClass["Sensitivity"])
  f1 <- ifelse((precision + recall) == 0, 0, (2 * precision * recall) / (precision + recall))

  list(
    accuracy = as.numeric(cm$overall["Accuracy"]),
    kappa = as.numeric(cm$overall["Kappa"]),
    precision = precision,
    recall = recall,
    f1 = f1,
    auc = auc_val
  )
}

model_registry <- list()
metrics_rows <- list()

add_result <- function(name, metrics, model_obj = NULL) {
  model_registry[[name]] <<- model_obj
  metrics_rows[[length(metrics_rows) + 1]] <<- tibble::tibble(
    model = name,
    accuracy = metrics$accuracy,
    kappa = metrics$kappa,
    precision = metrics$precision,
    recall = metrics$recall,
    f1 = metrics$f1,
    auc = metrics$auc
  )
}

# --------------------------
# ML model 1: Regularized Logistic Regression
# --------------------------
message("Training: Regularized Logistic Regression (glmnet)")
glm_fit <- caret::train(
  form,
  data = train_df,
  method = "glmnet",
  trControl = train_ctrl,
  metric = "ROC",
  tuneLength = 10
)
glm_prob <- predict(glm_fit, test_df, type = "prob")[, "yes"]
glm_metrics <- evaluate_binary(test_df[[target_col]], glm_prob)
add_result("LogReg_glmnet", glm_metrics, glm_fit)

# --------------------------
# ML model 2: Decision Tree
# --------------------------
message("Training: Decision Tree (rpart)")
rpart_fit <- caret::train(
  form,
  data = train_df,
  method = "rpart",
  trControl = train_ctrl,
  tuneLength = 10,
  metric = "ROC"
)
rpart_prob <- predict(rpart_fit, test_df, type = "prob")[, "yes"]
rpart_metrics <- evaluate_binary(test_df[[target_col]], rpart_prob)
add_result("RPART", rpart_metrics, rpart_fit)

# --------------------------
# ML model 3: Random Forest
# --------------------------
message("Training: Random Forest")
rf_fit <- caret::train(
  form,
  data = train_df,
  method = "rf",
  trControl = train_ctrl,
  tuneLength = 5,
  metric = "ROC",
  ntree = 200
)
rf_prob <- predict(rf_fit, test_df, type = "prob")[, "yes"]
rf_metrics <- evaluate_binary(test_df[[target_col]], rf_prob)
add_result("RandomForest", rf_metrics, rf_fit)

# --------------------------
# ML model 4: Neural Net (shallow, caret::nnet)
# --------------------------
message("Training: MLP (caret::nnet)")
nnet_fit <- caret::train(
  form,
  data = train_df,
  method = "nnet",
  trControl = train_ctrl,
  metric = "ROC",
  tuneLength = 5,
  preProcess = c("center", "scale"),
  trace = FALSE,
  maxit = 200
)
nnet_prob <- predict(nnet_fit, test_df, type = "prob")[, "yes"]
nnet_metrics <- evaluate_binary(test_df[[target_col]], nnet_prob)
add_result("MLP_nnet", nnet_metrics, nnet_fit)

# --------------------------
# ML model 5: XGBoost
# --------------------------
message("Training: XGBoost")
xgb_fit <- caret::train(
  form,
  data = train_df,
  method = "xgbTree",
  trControl = train_ctrl,
  metric = "ROC",
  tuneLength = 3
)
xgb_prob <- predict(xgb_fit, test_df, type = "prob")[, "yes"]
xgb_metrics <- evaluate_binary(test_df[[target_col]], xgb_prob)
add_result("XGBoost", xgb_metrics, xgb_fit)

# --------------------------
# Compare and select best model
# --------------------------
results_tbl <- dplyr::bind_rows(metrics_rows) |>
  dplyr::arrange(dplyr::desc(auc), dplyr::desc(f1), dplyr::desc(accuracy))

readr::write_csv(results_tbl, file.path(out_base, "summaries", "model_comparison.csv"))

best <- results_tbl[1, ]
best_name <- best$model[[1]]

best_summary_path <- file.path(out_base, "summaries", "best_model_summary.txt")
summary_lines <- c(
  paste0("Best model: ", best_name),
  paste0("AUC: ", round(best$auc[[1]], 5)),
  paste0("F1: ", round(best$f1[[1]], 5)),
  paste0("Accuracy: ", round(best$accuracy[[1]], 5)),
  "",
  "Ranking (top to bottom):",
  paste0(results_tbl$model, " [AUC=", round(results_tbl$auc, 5), ", F1=", round(results_tbl$f1, 5), ", Acc=", round(results_tbl$accuracy, 5), "]")
)
writeLines(summary_lines, con = best_summary_path)

best_obj <- model_registry[[best_name]]
saveRDS(best_obj, file.path(out_base, "models", "best_model.rds"))

message("Training and model selection complete.")
message("Best model: ", best_name)
message("Outputs saved under: ", out_base)
