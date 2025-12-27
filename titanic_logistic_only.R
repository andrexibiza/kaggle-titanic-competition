# ==============================================================================
# TITANIC LOGISTIC REGRESSION BENCHMARK
# Goal: Test if pure Logistic Regression (with interactions) can beat the Ensemble.
# ==============================================================================

library(caret)
library(dplyr)
library(stringr)
library(glmnet)

set.seed(42)

# ==============================================================================
# 1. DATA PREP (V4 Standard)
# ==============================================================================
cat("Loading Data...\n")
train <- read.csv("c:/Git/kaggle-titanic-competition/train.csv", stringsAsFactors = FALSE)
test  <- read.csv("c:/Git/kaggle-titanic-competition/test.csv",  stringsAsFactors = FALSE)
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

# V4 "Secret Sauce" (GroupSurvived)
full$Surname <- sapply(full$Name, function(x) strsplit(x, split = ",")[[1]][1])
full$FamilySurvived <- sapply(1:nrow(full), function(i) {
  pid <- full$PassengerId[i]; surname <- full$Surname[i]; fare <- full$Fare[i]
  family <- train[train$Surname == surname & train$PassengerId != pid & abs(train$Fare - fare) < 5, ]
  if (nrow(family) == 0) return(0.5)
  mean(family$Survived, na.rm = TRUE)
})
full$TicketSurvived <- sapply(1:nrow(full), function(i) {
  pid <- full$PassengerId[i]; ticket <- full$Ticket[i]
  group <- train[train$Ticket == ticket & train$PassengerId != pid, ]
  if (nrow(group) == 0) return(0.5)
  mean(group$Survived, na.rm = TRUE)
})
full$GroupSurvived <- pmax(full$FamilySurvived, full$TicketSurvived)

# ==============================================================================
# 2. LOGISTIC REGRESSION SPECIFIC: INTERACTIONS
# ==============================================================================
# Linear models can't find "Women in 3rd class" unless we tell them "Sex * Pclass".
model_data <- full %>% select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title, FamilySize, GroupSurvived)

# Create Dummy Variables explicitly to allow interactions
# Note: In R formulas, `Sex * Pclass` creates the interaction automatically.

train_df <- model_data[1:nrow(train), ]
train_df$Survived <- as.factor(ifelse(train$Survived == 1, "Yes", "No"))

# ==============================================================================
# 3. TRAINING
# ==============================================================================
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE)

cat("Training Baseline Logistic Regression (Main Effects only)...\n")
fit_glm_base <- train(
  Survived ~ ., 
  data = train_df, 
  method = "glm", 
  family = "binomial",
  trControl = ctrl
)

cat("Training Interaction Logistic Regression (Sex*Pclass + Age*Pclass)...\n")
fit_glm_int <- train(
  Survived ~ . + Sex:Pclass + Age:Pclass + GroupSurvived:Sex, 
  data = train_df, 
  method = "glm", 
  family = "binomial",
  trControl = ctrl
)

cat("\n--- RESULTS ---\n")
cat("GLM (Base) CV Accuracy:       ", max(fit_glm_base$results$Accuracy), "\n")
cat("GLM (Interactions) CV Accuracy:", max(fit_glm_int$results$Accuracy), "\n")
cat("V4 Ensemble Average CV:        ~0.83-0.84\n")

if (max(fit_glm_int$results$Accuracy) > 0.83) {
    cat("CONCLUSION: Logistic Regression is competitive!\n")
} else {
    cat("CONCLUSION: Logistic Regression underperforms Trees on this data.\n")
}
