# ==============================================================================
# TITANIC WCG MODEL (Chris Deotte Strategy)
# Base: V4 Ensemble (RF+XGB+GLM)
# Override: WCG Heuristic (Family Leakage)
# ==============================================================================

library(caret)
library(dplyr)
library(stringr)
library(xgboost)
library(ranger)

# Reproducibility
set.seed(42)

# ==============================================================================
# 1. LOAD & BASE MODEL (Re-implementing V4 logic for clean slate)
# ==============================================================================
cat("Loading data & Re-training V4 Base...\n")
train <- read.csv("c:/Git/kaggle-titanic-competition/train.csv", stringsAsFactors = FALSE)
test  <- read.csv("c:/Git/kaggle-titanic-competition/test.csv",  stringsAsFactors = FALSE)
test_ids <- test$PassengerId
test$Survived <- NA
full <- bind_rows(train, test)

# -- Basic V4 Feature Engineering --
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

# Prep for Models
full_clean <- full %>% select(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title, Deck, FamilySize, Survived)
train_df <- full_clean[1:nrow(train), ]
test_df  <- full_clean[(nrow(train) + 1):nrow(full), ]
test_df$Survived <- NULL
train_df$Survived <- as.factor(ifelse(train$Survived == 1, "Yes", "No"))

# Train V4 Ensemble (XGB, RF, GLM) - Simplified for speed/robustness
ctrl <- trainControl(method = "none", classProbs = TRUE)

# XGB
dummy <- dummyVars(~ ., data = train_df %>% select(-Survived))
xgb_x <- predict(dummy, newdata = train_df %>% select(-Survived))
xgb_test_x <- predict(dummy, newdata = test_df)

model_xgb <- caret::train(
  x = xgb_x, y = train_df$Survived,
  method = "xgbTree", trControl = ctrl,
  tuneGrid = expand.grid(nrounds=100, max_depth=3, eta=0.1, gamma=0, colsample_bytree=0.8, min_child_weight=1, subsample=0.8)
)

# RF
model_rf <- caret::train(
  Survived ~ ., data = train_df,
  method = "ranger", trControl = ctrl,
  tuneGrid = expand.grid(mtry=3, splitrule="gini", min.node.size=5)
)

# GLM
model_glm <- caret::train(
  Survived ~ ., data = train_df,
  method = "glmnet", trControl = ctrl,
  tuneGrid = expand.grid(alpha=0.5, lambda=0.01) # Simple elastic net
)

# Base Predictions
p_xgb <- predict(model_xgb, newdata = xgb_test_x, type = "prob")$Yes
p_rf  <- predict(model_rf, newdata = test_df, type = "prob")$Yes
p_glm <- predict(model_glm, newdata = test_df, type = "prob")$Yes

base_prob <- (p_xgb + p_rf + p_glm) / 3
base_pred <- ifelse(base_prob > 0.5, 1, 0) # 0/1 Vector

# ==============================================================================
# 2. WCG HEURISTIC (The "Secret Sauc"e)
# ==============================================================================
cat("Applying WCG Logic...\n")

# Use 'full' dataframe again to get Name/Ticket info
# Create Surnames
full$Surname <- sapply(full$Name, function(x) strsplit(x, split = ",")[[1]][1])

# 2.1 Group ID: Ticket + Surname
# WCG relies on unique groups. Ticket is best. Surname is backup.
# Helper function
full$GroupId <- paste0(full$Surname, "-", full$Pclass, "-", full$Ticket)

# 2.2 Calculate Group Survival Rates (TRAINING DATA ONLY)
# We map GroupId -> SurvivalRate
train_groups <- full[1:nrow(train), ]
group_survival <- train_groups %>%
  group_by(GroupId) %>%
  summarise(
    GroupCount = n(),
    Survivors = sum(Survived),
    Rate = mean(Survived)
  )

# 2.3 Apply Overrides to TEST Set
# Join rate to test passengers
test_wcg <- full[(nrow(train) + 1):nrow(full), ]
test_wcg <- test_wcg %>% 
  select(PassengerId, Sex, Age, Title, GroupId) %>%
  left_join(group_survival, by = "GroupId")

# Initialize Final Prediction with Base Model
final_pred_wcg <- base_pred 

# Counters
n_overrides <- 0

for (i in 1:length(final_pred_wcg)) {
  # Passenger details
  pid <- test_wcg$PassengerId[i]
  p_rate <- test_wcg$Rate[i]
  p_sex <- test_wcg$Sex[i]
  p_title <- as.character(test_wcg$Title[i])
  
  # "Boy" definition: Title is Master (Crucial!)
  is_boy <- (p_title == "Master")
  is_female <- (p_sex == "female")
  
  # Skip singletons or groups not in training set
  if (is.na(p_rate)) next
  
  # RULE 1: WOMAN-CHILD-SURVIVE
  # If 100% of group survived in train, and passenger is Woman or Boy -> Predict 1
  if ((is_female | is_boy) & p_rate == 1.0) {
    if (final_pred_wcg[i] == 0) {
      cat("Override: PID", pid, "-> 1 (WCG Survive Rule)\n")
      n_overrides <- n_overrides + 1
    }
    final_pred_wcg[i] <- 1
  }
  
  # RULE 2: WOMAN-CHILD-DIE
  # If 0% of group survived in train, and passenger is Woman or Boy -> Predict 0
  if ((is_female | is_boy) & p_rate == 0.0) {
    if (final_pred_wcg[i] == 1) {
      cat("Override: PID", pid, "-> 0 (WCG Die Rule)\n")
      n_overrides <- n_overrides + 1
    }
    final_pred_wcg[i] <- 0
  }
}

cat("\nTotal WCG Overrides applied:", n_overrides, "\n")

# ==============================================================================
# 3. SUBMISSION
# ==============================================================================
submission <- data.frame(PassengerId = test_ids, Survived = final_pred_wcg)
write.csv(submission, "c:/Git/kaggle-titanic-competition/submission_wcg.csv", row.names = FALSE)

cat("Submission saved to submission_wcg.csv\n")
cat("Done!\n")
