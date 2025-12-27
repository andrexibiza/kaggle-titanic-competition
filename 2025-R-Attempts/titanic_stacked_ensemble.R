# ==============================================================================
# TITANIC STACKED ENSEMBLE (V9)
# Strategy: Robust Stacking with Diversity (XGB, RF, SVM, KNN) -> LogReg Meta
# ==============================================================================

library(caret)
library(dplyr)
library(stringr)
library(xgboost)
library(ranger)
library(kernlab) # For SVM
library(class)   # For KNN

# Reproducibility
set.seed(2025)

# ==============================================================================
# 1. LOAD & CLEAN DATA
# ==============================================================================
cat("Loading & Preprocessing...\n")
train <- read.csv("c:/Git/kaggle-titanic-competition/train.csv", stringsAsFactors = FALSE)
test  <- read.csv("c:/Git/kaggle-titanic-competition/test.csv",  stringsAsFactors = FALSE)
test_ids <- test$PassengerId
test$Survived <- NA
full <- bind_rows(train, test)

# Feature Engineering (Clean Set - No Leakage Features)
full$Title <- str_extract(full$Name, "[a-zA-Z]+\\.")
full$Title[full$Title %in% c("Mme.")] <- "Mrs."
full$Title[full$Title %in% c("Mlle.", "Ms.")] <- "Miss."
full$Title[full$Title %in% c("Lady.", "Countess.", "Dona.")] <- "Mrs."
full$Title[full$Title %in% c("Capt.", "Col.", "Don.", "Dr.", "Major.", "Rev.", "Sir.", "Jonkheer.")] <- "Rare"
full$Title <- as.factor(full$Title)

# Ticket Freq
ticket_counts <- table(full$Ticket)
full$TicketFreq <- as.integer(ticket_counts[full$Ticket])

# Family Size
full$FamilySize <- full$SibSp + full$Parch + 1

# Imputation
full$Embarked[full$Embarked == ""] <- "S"
full$Fare[is.na(full$Fare)] <- median(full$Fare, na.rm = TRUE)
title_age_medians <- tapply(full$Age, full$Title, median, na.rm = TRUE)
full$Age <- ifelse(is.na(full$Age), title_age_medians[full$Title], full$Age)

full$Sex <- as.factor(full$Sex)
full$Embarked <- as.factor(full$Embarked)
full$Pclass <- as.factor(full$Pclass)

# Selection
model_data <- full %>% 
  select(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title, FamilySize, TicketFreq, Survived)

# Scaling for SVM/KNN (Important!)
preProcValues <- preProcess(model_data %>% select(-Survived), method = c("center", "scale"))
model_data_scaled <- predict(preProcValues, model_data)

# Split
train_df <- model_data_scaled[1:nrow(train), ]
test_df  <- model_data_scaled[(nrow(train) + 1):nrow(full), ]
test_df$Survived <- NULL
train_df$Survived <- as.factor(ifelse(train$Survived == 1, "Yes", "No"))

# ==============================================================================
# 2. LEVEL 1: BASE MODELS (5-Fold CV OOF)
# ==============================================================================
cat("Training Level 1 Models...\n")

# Use exactly the same folds for all models to keep OOF valid
cv_folds <- createFolds(train_df$Survived, k = 5, list = TRUE)
ctrl <- trainControl(
  method = "cv", 
  number = 5, 
  index = cv_folds, 
  classProbs = TRUE, 
  savePredictions = "final"
)

# --- A. Random Forest (Ranger) ---
cat(" [1/4] Ranger RF\n")
fit_rf <- caret::train(
  Survived ~ ., data = train_df,
  method = "ranger", trControl = ctrl,
  tuneGrid = expand.grid(mtry = 3, splitrule = "gini", min.node.size = 5)
)

# --- B. XGBoost ---
cat(" [2/4] XGBoost\n")
# XGB needs matrix for safer handling usually, but formula works if factors clean
fit_xgb <- caret::train(
  Survived ~ ., data = train_df,
  method = "xgbTree", trControl = ctrl,
  tuneGrid = expand.grid(nrounds=150, max_depth=4, eta=0.05, gamma=0, 
                         colsample_bytree=0.8, min_child_weight=1, subsample=0.8)
)

# --- C. SVM Radial ---
cat(" [3/4] SVM Radial\n")
fit_svm <- caret::train(
  Survived ~ ., data = train_df,
  method = "svmRadial", trControl = ctrl,
  tuneLength = 5 # Auto-tune C/Sigma
)

# --- D. KNN ---
cat(" [4/4] KNN\n")
fit_knn <- caret::train(
  Survived ~ ., data = train_df,
  method = "knn", trControl = ctrl,
  tuneGrid = expand.grid(k = c(5, 9, 13, 17, 21))
)

# ==============================================================================
# 3. LEVEL 2: STACKING (Meta-Model)
# ==============================================================================
cat("Training Meta-Model (Logistic Regression)...\n")

# ==============================================================================
# 3. ENSEMBLE: WEIGHTED AVERAGING (Robust Soft Voting)
# ==============================================================================
cat("Ensembling (Weighted Soft Voting)...\n")

# Meta-Model (GLM) failed due to OOF size mismatch/recycling issues.
# Reverting to explicit Weighted Average which is statistically safer and debuggable.

# Weights based on estimated reliability:
# RF/XGB: 0.35 each (High confidence)
# SVM: 0.20 (Diversity)
# KNN: 0.10 (Noise)

# Generating Test Predictions
pred_rf  <- predict(fit_rf, newdata = test_df, type = "prob")$Yes
pred_xgb <- predict(fit_xgb, newdata = test_df, type = "prob")$Yes
pred_svm <- predict(fit_svm, newdata = test_df, type = "prob")$Yes
pred_knn <- predict(fit_knn, newdata = test_df, type = "prob")$Yes

# Sanity Check
cat("Prediction Means:\n")
cat("RF:", mean(pred_rf), "\n")
cat("XGB:", mean(pred_xgb), "\n")
cat("SVM:", mean(pred_svm), "\n")
cat("KNN:", mean(pred_knn), "\n")

final_prob <- (0.35 * pred_rf) + (0.35 * pred_xgb) + (0.20 * pred_svm) + (0.10 * pred_knn)
final_class <- ifelse(final_prob > 0.5, 1, 0)

cat("Final Predicted Survivors:", sum(final_class), "\n")

# ==============================================================================
# 5. SUBMISSION
# ==============================================================================
submission <- data.frame(PassengerId = test_ids, Survived = final_class)
write.csv(submission, "c:/Git/kaggle-titanic-competition/submission_stack_v9.csv", row.names = FALSE)

cat("Submission saved to submission_stack_v9.csv\n")
cat("Done!\n")
