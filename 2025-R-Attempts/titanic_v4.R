# ==============================================================================
# TITANIC v4 - Without rule-based overrides (v3 hurt performance)
# Going back to v2 approach with better hyperparameters
# ==============================================================================

library(caret)
library(dplyr)
library(stringr)
library(xgboost)
library(ranger)

cat("Loading data...\n")
train <- read.csv("c:/Git/kaggle-titanic-competition/train.csv", stringsAsFactors = FALSE)
test  <- read.csv("c:/Git/kaggle-titanic-competition/test.csv",  stringsAsFactors = FALSE)

train_ids <- train$PassengerId
test_ids  <- test$PassengerId

test$Survived <- NA
full <- bind_rows(train, test)

# ==============================================================================
# Feature Engineering (Same as v2 - this worked)
# ==============================================================================
cat("Engineering features...\n")

# Title
full$Title <- str_extract(full$Name, "[a-zA-Z]+\\.")
full$Title[full$Title %in% c("Mme.")] <- "Mrs."
full$Title[full$Title %in% c("Mlle.", "Ms.")] <- "Miss."
full$Title[full$Title %in% c("Lady.", "Countess.", "Dona.")] <- "Mrs."
full$Title[full$Title %in% c("Capt.", "Col.", "Don.", "Dr.", "Major.", "Rev.", "Sir.", "Jonkheer.")] <- "Rare"
full$Title <- as.factor(full$Title)

# Surname  
full$Surname <- sapply(full$Name, function(x) strsplit(x, split = ",")[[1]][1])

# Family Size
full$FamilySize <- full$SibSp + full$Parch + 1
full$IsAlone <- as.factor(ifelse(full$FamilySize == 1, 1, 0))

# Deck
full$Deck <- ifelse(full$Cabin == "", "U", substr(full$Cabin, 1, 1))
full$Deck <- as.factor(full$Deck)

# Imputation
full$Embarked[full$Embarked == ""] <- "S"
full$Fare[is.na(full$Fare)] <- median(full$Fare, na.rm = TRUE)

title_age_medians <- tapply(full$Age, full$Title, median, na.rm = TRUE)
full$Age <- ifelse(is.na(full$Age), title_age_medians[full$Title], full$Age)

# Age Group
full$AgeGroup <- cut(full$Age, breaks = c(0, 12, 18, 35, 60, Inf), 
                     labels = c("Child", "Teen", "Adult", "MiddleAge", "Senior"), right = FALSE)

# Fare Group
full$FareGroup <- cut(full$Fare, breaks = c(-Inf, 7.91, 14.454, 31, Inf),
                      labels = c("Low", "MedLow", "MedHigh", "High"))

# Encoding
full$Sex <- as.factor(full$Sex)
full$Embarked <- as.factor(full$Embarked)
full$Pclass <- as.factor(full$Pclass)

# ==============================================================================
# Family/Ticket Survival Rate (Same as v2)
# ==============================================================================
cat("Computing group survival features...\n")

train$Surname <- sapply(train$Name, function(x) strsplit(x, split = ",")[[1]][1])

full$FamilySurvived <- sapply(1:nrow(full), function(i) {
  pid <- full$PassengerId[i]
  surname <- full$Surname[i]
  fare <- full$Fare[i]
  
  family <- train[train$Surname == surname & 
                    train$PassengerId != pid &
                    abs(train$Fare - fare) < 5, ]
  
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

# ==============================================================================
# Final Prep
# ==============================================================================
full_clean <- full %>% select(-PassengerId, -Name, -Ticket, -Cabin, -Surname)

train_final <- full_clean[1:nrow(train), ]
test_final  <- full_clean[(nrow(train) + 1):nrow(full), ]
test_final$Survived <- NULL

train_final$Survived <- as.factor(ifelse(train_final$Survived == 1, "Yes", "No"))

# ==============================================================================
# Train Final Models with LESS aggressive tuning (to avoid overfitting)
# ==============================================================================
cat("Training models...\n")

# Simple controls - avoid overfitting
myControl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  savePredictions = "final"
)

# XGBoost with conservative parameters
set.seed(42)
model_xgb <- train(
  Survived ~ ., 
  data = train_final,
  method = "xgbTree",
  trControl = myControl,
  tuneGrid = expand.grid(
    nrounds = 100,
    max_depth = 3, # Keep shallow to avoid overfitting
    eta = 0.1,
    gamma = 0,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    subsample = 0.8
  ),
  verbose = FALSE
)

cat("XGBoost CV Accuracy:", max(model_xgb$results$Accuracy), "\n")

# Random Forest
set.seed(42)
model_rf <- train(
  Survived ~ ., 
  data = train_final,
  method = "ranger",
  trControl = myControl,
  tuneGrid = expand.grid(mtry = 3, splitrule = "gini", min.node.size = 5),
  importance = "impurity"
)

cat("RF CV Accuracy:", max(model_rf$results$Accuracy), "\n")

# GLMnet (linear model as regularizer)
set.seed(42)
model_glm <- train(
  Survived ~ ., 
  data = train_final,
  method = "glmnet",
  trControl = myControl,
  preProcess = c("center", "scale"),
  tuneLength = 3
)

cat("GLMnet CV Accuracy:", max(model_glm$results$Accuracy), "\n")

# ==============================================================================
# Ensemble: Average all three models
# ==============================================================================
cat("Generating ensemble predictions...\n")

pred_xgb <- predict(model_xgb, newdata = test_final, type = "prob")$Yes
pred_rf  <- predict(model_rf, newdata = test_final, type = "prob")$Yes
pred_glm <- predict(model_glm, newdata = test_final, type = "prob")$Yes

# Simple average ensemble (more robust than weighted)
final_prob <- (pred_xgb + pred_rf + pred_glm) / 3
final_class <- ifelse(final_prob > 0.5, 1, 0)

# ==============================================================================
# Submission
# ==============================================================================
submission <- data.frame(PassengerId = test_ids, Survived = final_class)
write.csv(submission, "c:/Git/kaggle-titanic-competition/submission_v4.csv", row.names = FALSE)

cat("\nSubmission saved to submission_v4.csv\n")
cat("Done!\n")
