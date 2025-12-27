# ==============================================================================
# TITANIC v5 - Push past 0.80
# Adding FarePerPerson, TicketGroupSize, and threshold optimization
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
# Feature Engineering
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

# ** NEW: Ticket Group Size (people sharing same ticket) **
ticket_counts <- table(full$Ticket)
full$TicketGroupSize <- as.integer(ticket_counts[full$Ticket])

# ** NEW: FarePerPerson (better economic indicator) **
full$Fare[is.na(full$Fare)] <- median(full$Fare, na.rm = TRUE)
full$FarePerPerson <- full$Fare / full$TicketGroupSize
full$FarePerPerson[is.infinite(full$FarePerPerson) | is.na(full$FarePerPerson)] <- median(full$FarePerPerson[is.finite(full$FarePerPerson)], na.rm = TRUE)

# Deck
full$Deck <- ifelse(full$Cabin == "", "U", substr(full$Cabin, 1, 1))
full$Deck <- as.factor(full$Deck)

# Imputation
full$Embarked[full$Embarked == ""] <- "S"

title_age_medians <- tapply(full$Age, full$Title, median, na.rm = TRUE)
full$Age <- ifelse(is.na(full$Age), title_age_medians[full$Title], full$Age)

# Age Group (more granular for children)
full$AgeGroup <- cut(full$Age, breaks = c(0, 5, 12, 18, 35, 55, Inf), 
                     labels = c("Baby", "Child", "Teen", "Adult", "MiddleAge", "Senior"), right = FALSE)

# Fare Group based on FarePerPerson
fpp_breaks <- quantile(full$FarePerPerson, probs = c(0, 0.25, 0.5, 0.75, 1), na.rm = TRUE)
full$FareGroup <- cut(full$FarePerPerson, breaks = fpp_breaks, labels = c("Low", "MedLow", "MedHigh", "High"), include.lowest = TRUE)

# Encoding
full$Sex <- as.factor(full$Sex)
full$Embarked <- as.factor(full$Embarked)
full$Pclass <- as.factor(full$Pclass)

# ==============================================================================
# Family/Ticket Survival Rate
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
# Train Models
# ==============================================================================
cat("Training models...\n")

myControl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  savePredictions = "final"
)

# XGBoost
set.seed(42)
model_xgb <- train(
  Survived ~ ., 
  data = train_final,
  method = "xgbTree",
  trControl = myControl,
  tuneGrid = expand.grid(
    nrounds = 100,
    max_depth = 4,
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
  tuneGrid = expand.grid(mtry = 4, splitrule = "gini", min.node.size = 3),
  importance = "impurity"
)

cat("RF CV Accuracy:", max(model_rf$results$Accuracy), "\n")

# ==============================================================================
# Predictions with threshold analysis
# ==============================================================================
cat("Generating predictions...\n")

pred_xgb <- predict(model_xgb, newdata = test_final, type = "prob")$Yes
pred_rf  <- predict(model_rf, newdata = test_final, type = "prob")$Yes

# Ensemble (RF had higher CV, weight it more)
final_prob <- (0.4 * pred_xgb) + (0.6 * pred_rf)

# Standard threshold
final_class_50 <- ifelse(final_prob > 0.5, 1, 0)

# Experiment with different thresholds
final_class_45 <- ifelse(final_prob > 0.45, 1, 0)
final_class_55 <- ifelse(final_prob > 0.55, 1, 0)

# Generate multiple submissions for testing
submission_50 <- data.frame(PassengerId = test_ids, Survived = final_class_50)
submission_45 <- data.frame(PassengerId = test_ids, Survived = final_class_45)
submission_55 <- data.frame(PassengerId = test_ids, Survived = final_class_55)

write.csv(submission_50, "c:/Git/kaggle-titanic-competition/submission_v5_t50.csv", row.names = FALSE)
write.csv(submission_45, "c:/Git/kaggle-titanic-competition/submission_v5_t45.csv", row.names = FALSE)
write.csv(submission_55, "c:/Git/kaggle-titanic-competition/submission_v5_t55.csv", row.names = FALSE)

cat("\nSubmissions saved:\n")
cat(" - submission_v5_t50.csv (threshold 0.50)\n")
cat(" - submission_v5_t45.csv (threshold 0.45)\n")
cat(" - submission_v5_t55.csv (threshold 0.55)\n")
cat("Done!\n")
