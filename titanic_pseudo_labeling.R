# ==============================================================================
# TITANIC V10: PSEUDO-LABELING
# Strategy: Train V4 -> Select High Confidence Test Predictions -> Retrain V4
# ==============================================================================

library(caret)
library(dplyr)
library(stringr)
library(xgboost)
library(ranger)

set.seed(42)

# ==============================================================================
# 1. DATA LOADING & FEATURE ENGINEERING (Standard V4 Pipeline)
# ==============================================================================
cat("Loading & Preprocessing...\n")
train <- read.csv("c:/Git/kaggle-titanic-competition/train.csv", stringsAsFactors = FALSE)
test  <- read.csv("c:/Git/kaggle-titanic-competition/test.csv",  stringsAsFactors = FALSE)
test_ids <- test$PassengerId
test$Survived <- NA
full <- bind_rows(train, test)

# Features
full$Title <- str_extract(full$Name, "[a-zA-Z]+\\.")
full$Title[full$Title %in% c("Mme.")] <- "Mrs."
full$Title[full$Title %in% c("Mlle.", "Ms.")] <- "Miss."
full$Title[full$Title %in% c("Lady.", "Countess.", "Dona.")] <- "Mrs."
full$Title[full$Title %in% c("Capt.", "Col.", "Don.", "Dr.", "Major.", "Rev.", "Sir.", "Jonkheer.")] <- "Rare"
full$Title <- as.factor(full$Title)

full$Deck <- ifelse(full$Cabin == "", "U", substr(full$Cabin, 1, 1))
full$Deck <- as.factor(full$Deck)

full$FamilySize <- full$SibSp + full$Parch + 1
full$Embarked[full$Embarked == ""] <- "S"
full$Fare[is.na(full$Fare)] <- median(full$Fare, na.rm = TRUE)
title_age_medians <- tapply(full$Age, full$Title, median, na.rm = TRUE)
full$Age <- ifelse(is.na(full$Age), title_age_medians[full$Title], full$Age)

full$Sex <- as.factor(full$Sex)
full$Embarked <- as.factor(full$Embarked)
full$Pclass <- as.factor(full$Pclass)

# Model Data
full_clean <- full %>% select(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title, Deck, FamilySize, Survived)
train_df <- full_clean[1:nrow(train), ]
test_df  <- full_clean[(nrow(train) + 1):nrow(full), ]
test_df$Survived <- NULL
train_df$Survived <- as.factor(ifelse(train$Survived == 1, "Yes", "No"))

# ==============================================================================
# 2. STAGE 1: INITIAL MODEL TRAINING (V4)
# ==============================================================================
cat("\n--- STAGE 1: Initial Training ---\n")

ctrl <- trainControl(method = "none", classProbs = TRUE)

# XGB
dummy <- dummyVars(~ ., data = train_df %>% select(-Survived))
xgb_train_x <- predict(dummy, newdata = train_df %>% select(-Survived))
xgb_test_x  <- predict(dummy, newdata = test_df)

fit_xgb_1 <- caret::train(
  x = xgb_train_x, y = train_df$Survived,
  method = "xgbTree", trControl = ctrl,
  tuneGrid = expand.grid(nrounds=100, max_depth=3, eta=0.1, gamma=0, 
                         colsample_bytree=0.8, min_child_weight=1, subsample=0.8)
)

# RF
fit_rf_1 <- caret::train(
  Survived ~ ., data = train_df,
  method = "ranger", trControl = ctrl,
  tuneGrid = expand.grid(mtry=3, splitrule="gini", min.node.size=5),
  importance = "impurity"
)

# GLM
fit_glm_1 <- caret::train(
  Survived ~ ., data = train_df,
  method = "glmnet", trControl = ctrl,
  tuneGrid = expand.grid(alpha=0.5, lambda=0.01),
  preProcess = c("center", "scale")
)

# Stage 1 Predictions
p1_xgb <- predict(fit_xgb_1, newdata = xgb_test_x, type = "prob")$Yes
p1_rf  <- predict(fit_rf_1, newdata = test_df, type = "prob")$Yes
p1_glm <- predict(fit_glm_1, newdata = test_df, type = "prob")$Yes

prob_s1 <- (p1_xgb + p1_rf + p1_glm) / 3

# ==============================================================================
# 3. PSEUDO-LABELING
# ==============================================================================
cat("\n--- Selecting Pseudo-Labels ---\n")

# STRICT Thresholds
high_conf_idx <- which(prob_s1 > 0.99 | prob_s1 < 0.01)

if(length(high_conf_idx) == 0) {
  # Fallback to slightly looser if 0.99 yields nothing
  cat("Strict 0.99 threshold yielded 0 rows. Correcting to 0.95/0.05...\n")
  high_conf_idx <- which(prob_s1 > 0.95 | prob_s1 < 0.05)
}

cat("Found", length(high_conf_idx), "High Confidence Predictions.\n")

# Create Pseudo-Train Data
pseudo_test <- test_df[high_conf_idx, ]
pseudo_preds <- ifelse(prob_s1[high_conf_idx] > 0.5, "Yes", "No")
pseudo_test$Survived <- as.factor(pseudo_preds)

# Combine
train_augmented <- bind_rows(train_df, pseudo_test)

cat("Original Train Size:", nrow(train_df), "\n")
cat("Augmented Train Size:", nrow(train_augmented), "\n")

# ==============================================================================
# 4. STAGE 2: RETRAINING
# ==============================================================================
cat("\n--- STAGE 2: Retraining on Augmented Data ---\n")

# XGB (Re-do dummies for new size)
dummy_aug <- dummyVars(~ ., data = train_augmented %>% select(-Survived))
xgb_aug_x <- predict(dummy_aug, newdata = train_augmented %>% select(-Survived)) # Might differ slightly
xgb_aug_y <- train_augmented$Survived
xgb_test_x_idx <- predict(dummy_aug, newdata = test_df) # Ensure feature match

fit_xgb_2 <- caret::train(
  x = xgb_aug_x, y = xgb_aug_y,
  method = "xgbTree", trControl = ctrl,
  tuneGrid = expand.grid(nrounds=100, max_depth=3, eta=0.1, gamma=0, 
                         colsample_bytree=0.8, min_child_weight=1, subsample=0.8)
)

fit_rf_2 <- caret::train(
  Survived ~ ., data = train_augmented,
  method = "ranger", trControl = ctrl,
  tuneGrid = expand.grid(mtry=3, splitrule="gini", min.node.size=5)
)

fit_glm_2 <- caret::train(
  Survived ~ ., data = train_augmented,
  method = "glmnet", trControl = ctrl,
  tuneGrid = expand.grid(alpha=0.5, lambda=0.01),
  preProcess = c("center", "scale")
)

# ==============================================================================
# 5. FINAL PREDICTION
# ==============================================================================
cat("\nGenerating Final Predictions...\n")

p2_xgb <- predict(fit_xgb_2, newdata = xgb_test_x_idx, type = "prob")$Yes
p2_rf  <- predict(fit_rf_2, newdata = test_df, type = "prob")$Yes
p2_glm <- predict(fit_glm_2, newdata = test_df, type = "prob")$Yes

prob_final <- (p2_xgb + p2_rf + p2_glm) / 3
class_final <- ifelse(prob_final > 0.5, 1, 0)

# Submission
submission <- data.frame(PassengerId = test_ids, Survived = class_final)
write.csv(submission, "c:/Git/kaggle-titanic-competition/submission_pseudo_v10.csv", row.names = FALSE)

cat("Submission saved to submission_pseudo_v10.csv\n")
cat("Done!\n")
