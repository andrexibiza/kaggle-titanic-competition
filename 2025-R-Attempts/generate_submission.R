
# Load required libraries
library(caret)
library(dplyr)
library(stringr)
library(randomForest)
library(xgboost) # explicit load for caret
library(ranger)

# ==============================================================================
# 1. Data Loading & Preparation (Identical to Benchmark)
# ==============================================================================
cat("Loading data...\n")
train <- read.csv("c:/Git/kaggle-titanic-competition/train.csv", stringsAsFactors = FALSE)
test  <- read.csv("c:/Git/kaggle-titanic-competition/test.csv",  stringsAsFactors = FALSE)
test_ids <- test$PassengerId

test$Survived <- NA
full <- bind_rows(train, test)

# ==============================================================================
# 2. Feature Engineering
# ==============================================================================
cat("Engineering features...\n")

# Title
full$Title <- str_extract(full$Name, "[a-zA-Z]+\\.")
full$Title[full$Title %in% c("Mme.", "Mlle.")] <- "Miss."
full$Title[full$Title %in% c("Lady.", "the Countess.", "Capt.", "Col.", "Don.", "Dr.", "Major.", "Rev.", "Sir.", "Jonkheer.", "Dona.")] <- "Rare"
full$Title[full$Title == "Ms."] <- "Miss." 
full$Title <- as.factor(full$Title)

# Family Size
full$FamilySize <- full$SibSp + full$Parch + 1
full$IsAlone <- ifelse(full$FamilySize == 1, 1, 0)
full$IsAlone <- as.factor(full$IsAlone)

# Deck
full$Deck <- ifelse(full$Cabin == "", "U", substr(full$Cabin, 1, 1))
full$Deck <- as.factor(full$Deck)

# Imputation
full$Embarked[full$Embarked == ""] <- "S"
full$Fare[is.na(full$Fare)] <- median(full$Fare, na.rm = TRUE)

title_age_medians <- tapply(full$Age, full$Title, median, na.rm = TRUE)
fill_age <- function(age, title) {
  if (is.na(age)) return(title_age_medians[title]) else return(age)
}
full$Age <- mapply(fill_age, full$Age, full$Title)

# Encoding
full$Sex <- as.factor(full$Sex)
full$Embarked <- as.factor(full$Embarked)
full$Pclass <- as.factor(full$Pclass)

# Drop High Cardinality / Unused
full <- full %>% select(-PassengerId, -Name, -Ticket, -Cabin)

# Split back
train_clean <- full[1:nrow(train), ]
test_clean  <- full[(nrow(train) + 1):nrow(full), ]
test_clean$Survived <- NULL

# Target
train_clean$Survived <- as.factor(ifelse(train_clean$Survived == 1, "Yes", "No"))

# ==============================================================================
# 3. Final Model Training & Ensemble
# ==============================================================================
cat("Training final models on full dataset...\n")

control <- trainControl(method = "none", classProbs = TRUE) # No CV, just train

# XGBoost
set.seed(42)
final_xgb <- train(
  Survived ~ ., 
  data = train_clean,
  method = "xgbTree",
  trControl = control,
  tuneGrid = expand.grid(
    nrounds = 100, 
    max_depth = 3, 
    eta = 0.3, 
    gamma = 0, 
    colsample_bytree = 0.8, 
    min_child_weight = 1, 
    subsample = 1
  ) # Simplified grid or copy from best tune if known. Using robust defaults.
)

# Random Forest
set.seed(42)
final_rf <- train(
  Survived ~ ., 
  data = train_clean,
  method = "ranger",
  trControl = control,
  importance = "impurity",
  tuneGrid = expand.grid(mtry = 2, splitrule = "gini", min.node.size = 1)
)

# Predictions (Probabilities)
cat("Predicting...\n")
pred_xgb <- predict(final_xgb, newdata = test_clean, type = "prob")
pred_rf  <- predict(final_rf, newdata = test_clean, type = "prob")

# Ensemble (Weighted Average: XGB performed better, give it slightly more weight? Or equal?)
# XGB Acc: 0.84, RF Acc: 0.83. Let's do 0.6 XGB + 0.4 RF
final_prob <- (0.6 * pred_xgb$Yes) + (0.4 * pred_rf$Yes)
final_class <- ifelse(final_prob > 0.5, 1, 0)

# ==============================================================================
# 4. Submission
# ==============================================================================
submission <- data.frame(PassengerId = test_ids, Survived = final_class)
write.csv(submission, "c:/Git/kaggle-titanic-competition/submission_ensemble.csv", row.names = FALSE)

cat("Submission saved to submission_ensemble.csv\n")
