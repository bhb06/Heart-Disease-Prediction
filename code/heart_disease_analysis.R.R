# ============================
# 1. Load Libraries
# ============================
library(data.table)
library(ggplot2)
library(ggthemes)
library(corrplot)
library(caret)
library(randomForest)
library(xgboost)
library(e1071)
library(naniar)
library(dplyr)
library(tidyr)
library(forcats)
library(class)
library(viridis)
library(pROC)
library(smotefamily)


# ============================
# 2. Data Loading
# ============================
df <- read.csv("heart_disease.csv", stringsAsFactors = FALSE)
head(df)
str(df)
dim(df)
summary(df)


# ============================
# 3. Data Cleaning
# ============================
df[df == ""] <- NA
df[df == "None"] <- NA
colSums(is.na(df))

# Visualize missing values
gg_miss_var(df) + ggtitle("Visualization of Missing Data")

# Replace NA in Alcohol Consumption with mode
mode_value <- names(sort(table(df$`Alcohol.Consumption`), decreasing = TRUE))[1]
df$`Alcohol.Consumption`[is.na(df$'Alcohol.Consumption')] <- mode_value

# Remove remaining NA rows
df <- na.omit(df)
colSums(is.na(df))


# ============================
# 4. Exploratory Data Analysis (EDA)
# ============================
# Correlation matrix
numerical_vars <- df[sapply(df, is.numeric)]
correlation_matrix <- cor(numerical_vars, use = "complete.obs")
color_scale <- colorRampPalette(c("red", "white", "blue"))(200)
corrplot(correlation_matrix, method = "color", col = color_scale, tl.cex = 0.75, number.cex = 0.75, cl.cex = 1.5, addCoef.col = "black", tl.col = "black", tl.srt = 45)

# Histograms
melted_df <- numerical_vars %>% pivot_longer(everything())
ggplot(melted_df, aes(x = value)) + geom_histogram(bins = 20, fill = "steelblue", alpha = 0.7) + facet_wrap(~name, scales = "free") + theme_minimal() + ggtitle("Histograms of Numerical Data")

# Bar plots for categorical variables
categorical_cols <- df %>% select(where(is.character))
melted_df <- categorical_cols %>% pivot_longer(everything(), names_to = "Category", values_to = "Value")
ggplot(melted_df, aes(x = Value)) + geom_bar(fill = "steelblue", width = 0.7) + facet_wrap(~Category, scales = "free", ncol = 4) + theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1), strip.text = element_text(size = 6, face = "bold")) + labs(title = "Distribution of Categorical Variables", x = "", y = "Count")

# Boxplot: Age vs Heart Disease Status
ggplot(df, aes(x = as.factor(`Heart.Disease.Status`), y = Age, fill = as.factor(`Heart.Disease.Status`))) +
  geom_boxplot(alpha = 1) +
  scale_fill_manual(values = c("steelblue", "orange")) +
  theme_minimal() +
  labs(title = "Relationship between Age and Heart Disease Status", x = "Heart Disease Status", y = "Age") +
  theme(legend.position = "none")

# Mean values grouped by Heart Disease Status
heart_disease_grouped <- df %>% 
  group_by(`Heart.Disease.Status`) %>% 
  summarise(across(where(is.numeric), mean, na.rm = TRUE)) %>% 
  pivot_longer(cols = -`Heart.Disease.Status`, names_to = "Variable", values_to = "Mean")

num_variables <- length(unique(heart_disease_grouped$Variable))
color_palette <- scales::hue_pal()(num_variables)

ggplot(heart_disease_grouped, aes(x = `Heart.Disease.Status`, y = Mean, fill = Variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  scale_fill_manual(values = color_palette) +
  labs(title = "Mean Values of Numerical Variables Based on Heart Disease Status", x = "Heart Disease Status", y = "Mean Value", fill = "Variable") +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))

# Pie chart for Heart Disease Status
disease_counts <- data.frame(Status = names(table(df$`Heart.Disease.Status`)), Count = as.vector(table(df$`Heart.Disease.Status`)))
colnames(disease_counts) <- c("Status", "Count")

ggplot(disease_counts, aes(x = "", y = Count, fill = Status)) +
  geom_bar(stat = "identity", width = 1, color = "black") +
  coord_polar(theta = "y", start = pi/2) +
  scale_fill_manual(values = c("steelblue", "orange")) +
  theme_void() +
  labs(title = "Distribution of Heart Disease Status", fill = "Status") +
  geom_text(aes(label = paste0(round((Count/sum(Count))*100, 1), "%")), position = position_stack(vjust = 0.5), color = "white", size = 5)


# ============================
# 5. Feature Encoding
# ============================
label_mappings <- list()
for (col in names(df)) {
  if (is.factor(df[[col]]) || is.character(df[[col]])) {
    df[[col]] <- as.factor(df[[col]])
    levels_map <- levels(df[[col]])
    label_mappings[[col]] <- setNames(seq(0, length(levels_map) - 1), levels_map)
    df[[col]] <- as.numeric(df[[col]]) - 1
  }
}

# ============================
# 6. Data Splitting & Scaling
# ============================
X <- df[, !names(df) %in% 'Heart.Disease.Status']
y <- as.factor(df$`Heart.Disease.Status`)
set.seed(123)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
y_train <- y[trainIndex]
X_test <- X[-trainIndex, ]
y_test <- y[-trainIndex]

# Standardize
X_train <- scale(X_train)
X_test <- scale(X_test, center = attr(X_train, "scaled:center"), scale = attr(X_train, "scaled:scale"))
cat("Means of each feature:\n"); print(colMeans(X_train))
cat("\nStandard deviations of each feature:\n"); print(apply(X_train, 2, sd))


# ============================
# 7. SMOTE Oversampling
# ============================
train_data <- data.frame(X_train, Heart.Disease.Status = y_train)
smote_output <- SMOTE(X = train_data[, -ncol(train_data)], target = train_data$Heart.Disease.Status, K = 5)
train_data_smote <- smote_output$data
X_train_smote <- train_data_smote[, -ncol(train_data_smote)]
y_train_smote <- as.factor(train_data_smote$class)


# ============================
# 8. Evaluation Function
# ============================
evaluate_model <- function(predictions, probs, true_labels, model_name = "") {
  predictions <- as.factor(predictions)
  true_labels <- as.factor(true_labels)
  cm <- confusionMatrix(predictions, true_labels, positive = "1")
  roc_obj <- roc(as.numeric(true_labels), as.numeric(probs))
  auc_value <- auc(roc_obj)
  cat("\n---", model_name, "---\n")
  print(cm)
  cat("AUC:", round(auc_value, 4), "\n\n")
  return(list(Accuracy = cm$overall['Accuracy'], Precision = cm$byClass['Precision'], Recall = cm$byClass['Recall'], F1 = cm$byClass['F1'], AUC = auc_value))
}


# ============================
# 9. Feature Selection (Each model alone)
# ============================

# KNN Feature Selection
X_train_smote_df <- as.data.frame(X_train_smote)
y_train_smote_vec <- factor(y_train_smote, levels = c(0, 1))
nzv <- nearZeroVar(X_train_smote_df) # Remove near-zero variance predictors
if (length(nzv) > 0) {
  X_train_smote_df <- X_train_smote_df[, -nzv]
  cat("Removed", length(nzv), "near-zero variance features.\n")
}
control <- rfeControl(functions = caretFuncs, method = "cv", number = 5, verbose = TRUE) # build rfeControl with working dummy importance
control$functions$rank <- function(...) {
  data.frame(var = colnames(X_train_smote_df), Overall = 1)
}
set.seed(123)
knn_rfe <- rfe( # Run RFE
  x = X_train_smote_df,
  y = y_train_smote_vec,
  sizes = c(5, 10, 20),
  rfeControl = control,
  method = "knn"
)
selected_features_knn <- predictors(knn_rfe)
cat("Selected features for KNN from RFE:\n")
print(selected_features_knn)

# XGBoost Feature Selection
# STEP 1: Retrain XGBoost on all SMOTE-balanced data (if not already trained)
dtrain <- xgb.DMatrix(data = as.matrix(X_train_smote), label = as.numeric(y_train_smote) - 1)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = as.numeric(y_test) - 1)
xgb_model <- xgboost::xgboost(
  data = dtrain,
  objective = "binary:logistic",
  nrounds = 100,
  verbose = 0
)
# STEP 2: Extract top 10 important features
xgb_importance <- xgb.importance(model = xgb_model)
print(xgb_importance)
top_features_xgb <- xgb_importance$Feature[1:10]
cat("\nTop 10 Important Features for XGBoost:\n")
print(top_features_xgb)

# Logistic Regression Feature Selection
logistic_model <- glm(y_train_smote ~ ., data = as.data.frame(X_train_smote), family = binomial)
coef_summary <- summary(logistic_model)$coefficients[-1, , drop = FALSE]
coef_df <- data.frame(
  Feature = rownames(coef_summary),
  Estimate = coef_summary[, "Estimate"],
  Std.Error = coef_summary[, "Std. Error"],
  z.value = coef_summary[, "z value"],
  p.value = coef_summary[, "Pr(>|z|)"]
)
significant_features <- coef_df %>%
  filter(p.value < 0.05) %>%
  arrange(desc(abs(Estimate)))
selected_features_logreg <- significant_features$Feature
cat("\nSelected Features for Logistic Regression:\n")
print(selected_features_logreg)

# Random Forest Feature Selection
rf_model <- randomForest(x = X_train_smote, y = y_train_smote, ntree = 100, mtry = 3, importance = TRUE)
importance_values <- importance(rf_model)[, "MeanDecreaseGini"]
importance_df <- data.frame(
  Feature = names(importance_values),
  Importance = round(importance_values, 2)
) %>% arrange(desc(Importance))
selected_features_rf <- importance_df$Feature[1:10]
cat("\nTop 10 Important Features for Random Forest:\n")
print(selected_features_rf)


# ============================
# 10. KNN Classification
# ============================
knn_predictions <- knn(train = X_train_knn, test = X_test_knn, cl = y_train_smote, k = 5)
knn_probs <- as.numeric(knn_predictions)

# Evaluate and store metrics
knn_metrics <- evaluate_model(knn_predictions, knn_probs, y_test, "KNN (Selected Features)")
knn_accuracy <- knn_metrics$Accuracy


# ============================
# 11. XGBoost
# ============================
# Subset data using selected features
X_train_xgb <- X_train_smote[, top_features_xgb]
X_test_xgb <- X_test[, top_features_xgb]

dtrain_xgb <- xgb.DMatrix(data = as.matrix(X_train_xgb), label = as.numeric(y_train_smote) - 1)
dtest_xgb <- xgb.DMatrix(data = as.matrix(X_test_xgb), label = as.numeric(y_test) - 1)

# Retrain XGBoost with selected features
xgb_model_selected <- xgboost(
  data = dtrain_xgb,
  objective = "binary:logistic",
  nrounds = 100,
  verbose = 0
)
xgb_preds_selected <- predict(xgb_model_selected, dtest_xgb)
xgb_labels_selected <- ifelse(xgb_preds_selected > 0.5, 1, 0)
xgb_metrics <- evaluate_model(xgb_labels_selected, xgb_preds_selected, y_test, "XGBoost (Selected Features)")
xgb_accuracy <- xgb_metrics$Accuracy

# ============================
# 12. Logistic Regression
# ============================

# Subset SMOTE-balanced training and original test sets using selected features
X_train_selected_logreg <- X_train_smote[, selected_features_logreg]
X_test_selected_logreg <- X_test[, selected_features_logreg]

# Retrain logistic regression model with selected features
logistic_model_selected <- glm(y_train_smote ~ ., data = data.frame(X_train_selected_logreg, y_train_smote), family = binomial)

# Predict and evaluate
logistic_probs_selected <- predict(logistic_model_selected, as.data.frame(X_test_selected_logreg), type = "response")
logistic_predictions_selected <- ifelse(logistic_probs_selected > 0.5, 1, 0)
logistic_accuracy_selected <- sum(logistic_predictions_selected == y_test) / length(y_test)
cat("Logistic Regression Accuracy (Significant Features Only):", logistic_accuracy_selected, "\n")

# Evaluate the logistic regression model with significant features
evaluate_model(logistic_predictions_selected, logistic_probs_selected, y_test, "Logistic Regression (Significant Features Only)")


# ============================
# 13. Random Forest
# ============================

# Subset training and test sets using selected Random Forest features
X_train_selected_rf <- X_train_smote[, selected_features_rf]
X_test_selected_rf <- X_test[, selected_features_rf]

# Train Random Forest model with selected features
rf_model_selected <- randomForest(x = X_train_selected_rf, y = y_train_smote, ntree = 100, mtry = 3)

# Predict on the test set
rf_predictions_selected <- predict(rf_model_selected, X_test_selected_rf)

# Compute accuracy
rf_accuracy_selected <- sum(rf_predictions_selected == y_test) / length(y_test)
cat("Random Forest Accuracy (Selected Features): ", rf_accuracy_selected, "\n")

# Predict probabilities and evaluate
rf_probs_selected <- predict(rf_model_selected, X_test_selected_rf, type = "prob")[,2]
evaluate_model(rf_predictions_selected, rf_probs_selected, y_test, "Random Forest (Selected Features)")



# ============================
# 14. Model Accuracy Comparison Plot
# ============================
# Visual comparison of model accuracies using a bar plot

models <- c('KNN', 'XGBoost', 'Logistic Regression', 'Random Forest')
accuracies <- c(knn_accuracy, xgb_accuracy, logistic_accuracy_selected, rf_accuracy_selected)

# Create data frame for the plot
df_plot <- data.frame(models, accuracies)

# Plot all model accuracies
ggplot(df_plot, aes(x = models, y = accuracies, fill = models)) +
  geom_bar(stat = 'identity', show.legend = FALSE) +
  scale_fill_viridis(discrete = TRUE) + 
  labs(x = 'Models', y = 'Accuracy', title = 'Comparison of Model Accuracies') +
  ylim(0, 1) +
  geom_text(aes(label = sprintf('%.2f', accuracies)), vjust = -0.5) + 
  theme_minimal()
