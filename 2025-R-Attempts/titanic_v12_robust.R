# ==============================================================================
# TITANIC V12: ROBUST FEATURE ENGINEERING
# Fixes "Zero Variance" issue by replacing 0.5 default with Demographic Baseline.
# ==============================================================================

library(caret)
library(dplyr)
library(stringr)
library(xgboost)
library(ranger)
library(glmnet)

set.seed(42)

# ==============================================================================
# 1. DATA PREP
# ==============================================================================
cat("Loading Data...\n")
train <- read.csv("c:/Git/kaggle-titanic-competition/train.csv", stringsAsFactors = FALSE)
test  <- read.csv("c:/Git/kaggle-titanic-competition/test.csv",  stringsAsFactors = FALSE)
test_ids <- test$PassengerId
test$Survived <- NA
full <- bind_rows(train, test)

# Basic Features
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

# ==============================================================================
# 2. ROBUST SURVIVAL RATE CALCULATION
# ==============================================================================
cat("Calculating Robust Group Features...\n")

# A. Calculate Baseline Demographic Rates (Sex + Pclass)
# This serves as the "Prior" probability for singletons
demo_rates <- train %>%
  group_by(Sex, Pclass) %>%
  summarise(DemoRate = mean(Survived), .groups = 'drop')

# Fix Type Mismatch for Join: Ensure Pclass is Factor to match 'full'
demo_rates$Pclass <- as.factor(demo_rates$Pclass)

print("Demographic Baselines:")
print(demo_rates)

# Join Baselines to Full Data
full <- full %>% left_join(demo_rates, by = c("Sex", "Pclass"))

# B. Calculate Leakage (Family/Ticket)
full$Surname <- sapply(full$Name, function(x) strsplit(x, split = ",")[[1]][1])

full$FamilySurvived <- sapply(1:nrow(full), function(i) {
  pid <- full$PassengerId[i]
  surname <- full$Surname[i]
  fare <- full$Fare[i]
  # Find family in TRAIN
  family <- train[train$Surname == surname & train$PassengerId != pid & abs(train$Fare - fare) < 5, ]
  
  if (nrow(family) == 0) return(NA) # CHANGE: Return NA instead of 0.5
  mean(family$Survived, na.rm = TRUE)
})

full$TicketSurvived <- sapply(1:nrow(full), function(i) {
  pid <- full$PassengerId[i]
  ticket <- full$Ticket[i]
  # Find ticket group in TRAIN
  group <- train[train$Ticket == ticket & train$PassengerId != pid, ]
  
  if (nrow(group) == 0) return(NA) # CHANGE: Return NA instead of 0.5
  mean(group$Survived, na.rm = TRUE)
})

# C. Coalesce Logic
# 1. If Ticket info exists -> Use TicketSurvived
# 2. Else If Family info exists -> Use FamilySurvived
# 3. Else -> Use DemoRate (Baseline)
full$GroupProb <- coalesce(full$TicketSurvived, full$FamilySurvived, full$DemoRate)

# Sanity Check
cat("Checking variances:\n")
cat("Variance of GroupProb:", var(full$GroupProb, na.rm=TRUE), "\n") # Should be > 0

# ==============================================================================
# 3. MODELING (V4 Pipeline)
# ==============================================================================
cat("Training Models...\n")

full_clean <- full %>% select(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title, Deck, FamilySize, GroupProb, Survived)
train_final <- full_clean[1:nrow(train), ]
test_final  <- full_clean[(nrow(train) + 1):nrow(full), ]
test_final$Survived <- NULL
train_final$Survived <- as.factor(ifelse(train$Survived == 1, "Yes", "No"))

ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, savePredictions = "final")

# 1. XGB
set.seed(42)
model_xgb <- caret::train(
  Survived ~ ., data = train_final, method = "xgbTree", trControl = ctrl,
  tuneGrid = expand.grid(nrounds=100, max_depth=3, eta=0.1, gamma=0, 
                         colsample_bytree=0.8, min_child_weight=1, subsample=0.8)
)

# 2. RF
set.seed(42)
model_rf <- caret::train(
  Survived ~ ., data = train_final, method = "ranger", trControl = ctrl,
  tuneGrid = expand.grid(mtry=3, splitrule="gini", min.node.size=5),
  importance = "impurity"
)

# 3. GLM
set.seed(42)
model_glm <- caret::train(
  Survived ~ ., data = train_final, method = "glmnet", trControl = ctrl,
  tuneGrid = expand.grid(alpha=0.5, lambda=0.01),
  preProcess = c("center", "scale")
)

cat("CV Accuracies:\n")
cat("XGB:", max(model_xgb$results$Accuracy), "\n")
cat("RF:", max(model_rf$results$Accuracy), "\n")
cat("GLM:", max(model_glm$results$Accuracy), "\n")

# ==============================================================================
# 4. ENSEMBLE
# ==============================================================================
p_xgb <- predict(model_xgb, newdata = test_final, type = "prob")$Yes
p_rf  <- predict(model_rf, newdata = test_final, type = "prob")$Yes
p_glm <- predict(model_glm, newdata = test_final, type = "prob")$Yes

final_prob <- (p_xgb + p_rf + p_glm) / 3
final_class <- ifelse(final_prob > 0.5, 1, 0)

# Submission
submission <- data.frame(PassengerId = test_ids, Survived = final_class)
write.csv(submission, "c:/Git/kaggle-titanic-competition/submission_v12_robust.csv", row.names = FALSE)

cat("Submission saved to submission_v12_robust.csv\n")
