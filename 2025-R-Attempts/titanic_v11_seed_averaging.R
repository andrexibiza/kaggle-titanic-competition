# ==============================================================================
# TITANIC V11: SEED AVERAGING (STABILIZATION)
# Strategy: Run Champion V4 model across 20 different random seeds and average.
# Goal: Reduce variance to push score from 0.789 -> 0.80+
# ==============================================================================

library(caret)
library(dplyr)
library(stringr)
library(xgboost)
library(ranger)
library(glmnet)

# ==============================================================================
# 1. DATA PREP (IDENTICAL TO V4)
# ==============================================================================
cat("Loading and Prepping Data (V4 Pipeline)...\n")
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

# FSR/TSR (The "Secret Sauce")
full$Surname <- sapply(full$Name, function(x) strsplit(x, split = ",")[[1]][1])
full$FamilySurvived <- sapply(1:nrow(full), function(i) {
  pid <- full$PassengerId[i]
  surname <- full$Surname[i]
  fare <- full$Fare[i]
  family <- train[train$Surname == surname & train$PassengerId != pid & abs(train$Fare - fare) < 5, ]
  if (nrow(family) == 0) return(0.5)
  mean(family$Survived, na.rm = TRUE)
})
full$TicketSurvived <- sapply(1:nrow(full), function(i) {
  pid <- full$PassengerId[i]
  ticket <- full$Ticket[i]
  group <- train[train$Ticket == ticket & train$PassengerId != pid, ]
  if (nrow(group) == 0) return(0.5)
  mean(group$Survived, na.rm = TRUE)
})
full$GroupSurvived <- pmax(full$FamilySurvived, full$TicketSurvived)

# Final Data Objects
full_clean <- full %>% select(-PassengerId, -Name, -Ticket, -Cabin, -Surname)
train_final <- full_clean[1:nrow(train), ]
test_final  <- full_clean[(nrow(train) + 1):nrow(full), ]
test_final$Survived <- NULL
train_final$Survived <- as.factor(ifelse(train$Survived == 1, "Yes", "No"))

# ==============================================================================
# 2. SEED AVERAGING LOOP
# ==============================================================================
seeds <- c(42, 2023, 1234, 777, 888, 555, 111, 99, 101, 2025,
           33, 44, 55, 66, 77, 88, 90, 1000, 500, 1) # 20 Seeds! Needs to be robust.

cat("\nStarting Seed Averaging across", length(seeds), "seeds...\n")

predictions_matrix <- matrix(0, nrow = nrow(test_final), ncol = length(seeds))

# Control (CV off for speed in loop, we rely on the ensemble average)
ctrl <- trainControl(method = "none", classProbs = TRUE)

for (i in 1:length(seeds)) {
  s <- seeds[i]
  cat(sprintf("[%2d/%2d] Training with Seed: %d ... ", i, length(seeds), s))
  
  set.seed(s)
  
  # A. XGBoost
  model_xgb <- caret::train(
    Survived ~ ., data = train_final, method = "xgbTree", trControl = ctrl,
    tuneGrid = expand.grid(nrounds=100, max_depth=3, eta=0.1, gamma=0, 
                           colsample_bytree=0.8, min_child_weight=1, subsample=0.8)
  )
  
  # B. Random Forest
  model_rf <- caret::train(
    Survived ~ ., data = train_final, method = "ranger", trControl = ctrl,
    tuneGrid = expand.grid(mtry=3, splitrule="gini", min.node.size=5),
    importance = "impurity"
  )
  
  # C. GLMnet
  model_glm <- caret::train(
    Survived ~ ., data = train_final, method = "glmnet", trControl = ctrl,
    tuneGrid = expand.grid(alpha=0.5, lambda=0.01),
    preProcess = c("center", "scale")
  )
  
  # Predict
  p_xgb <- predict(model_xgb, newdata = test_final, type = "prob")$Yes
  p_rf  <- predict(model_rf, newdata = test_final, type = "prob")$Yes
  p_glm <- predict(model_glm, newdata = test_final, type = "prob")$Yes
  
  # Ensemble for this seed
  p_avg <- (p_xgb + p_rf + p_glm) / 3
  
  predictions_matrix[, i] <- p_avg
  cat("Done.\n")
}

# ==============================================================================
# 3. AGGREGATION & SUBMISSION
# ==============================================================================
cat("\nAggregating predictions...\n")

final_probs <- rowMeans(predictions_matrix)
final_preds <- ifelse(final_probs > 0.5, 1, 0)

cat("Final Predicted Survivors:", sum(final_preds), "\n")

submission <- data.frame(PassengerId = test_ids, Survived = final_preds)
write.csv(submission, "c:/Git/kaggle-titanic-competition/submission_v11_seed_avg.csv", row.names = FALSE)

cat("Submission saved to submission_v11_seed_avg.csv\n")
