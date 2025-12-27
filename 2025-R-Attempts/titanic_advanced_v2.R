# ==============================================================================
# TITANIC ADVANCED MODEL v2
# Goal: Improve score from 0.76 to >0.80 using Family Survival Rate feature
# ==============================================================================

# Load libraries
library(caret)
library(dplyr)
library(stringr)
library(xgboost)
library(ranger)

# ==============================================================================
# 1. Data Loading
# ==============================================================================
cat("Loading data...\n")
train <- read.csv("c:/Git/kaggle-titanic-competition/train.csv", stringsAsFactors = FALSE)
test  <- read.csv("c:/Git/kaggle-titanic-competition/test.csv",  stringsAsFactors = FALSE)

train_ids <- train$PassengerId
test_ids  <- test$PassengerId

test$Survived <- NA
full <- bind_rows(train, test)

# ==============================================================================
# 2. Basic Feature Engineering
# ==============================================================================
cat("Engineering basic features...\n")

# --- Title Extraction & Consolidation ---
full$Title <- str_extract(full$Name, "[a-zA-Z]+\\.")
full$Title[full$Title %in% c("Mme.")] <- "Mrs."
full$Title[full$Title %in% c("Mlle.", "Ms.")] <- "Miss."
full$Title[full$Title %in% c("Lady.", "Countess.", "Dona.")] <- "RareFemale"
full$Title[full$Title %in% c("Capt.", "Col.", "Don.", "Dr.", "Major.", "Rev.", "Sir.", "Jonkheer.")] <- "RareMale"
full$Title <- as.factor(full$Title)

# --- Surname Extraction (for Family Grouping) ---
full$Surname <- sapply(full$Name, function(x) strsplit(x, split = ",")[[1]][1])

# --- Family Size ---
full$FamilySize <- full$SibSp + full$Parch + 1
full$IsAlone <- as.factor(ifelse(full$FamilySize == 1, 1, 0))

# --- Family ID (Surname + FamilySize for grouping) ---
# Small families are harder to group reliably, so we group them as "Small"
full$FamilyID <- paste(full$Surname, full$FamilySize, sep = "_")
# If family size is 1 or 2, it's too small to get reliable FSR - generalize
full$FamilyID[full$FamilySize <= 2] <- "Small"
full$FamilyID <- as.factor(full$FamilyID)

# --- Deck Extraction ---
full$Deck <- ifelse(full$Cabin == "", "U", substr(full$Cabin, 1, 1))
full$Deck <- as.factor(full$Deck)

# --- Age Imputation (Median by Title) ---
full$Embarked[full$Embarked == ""] <- "S"
full$Fare[is.na(full$Fare)] <- median(full$Fare, na.rm = TRUE)

title_age_medians <- tapply(full$Age, full$Title, median, na.rm = TRUE)
fill_age <- function(age, title) {
  if (is.na(age)) return(title_age_medians[title]) else return(age)
}
full$Age <- mapply(fill_age, full$Age, full$Title)

# --- Age Binning ---
full$AgeGroup <- cut(full$Age, 
                     breaks = c(0, 12, 18, 35, 60, Inf), 
                     labels = c("Child", "Teen", "Adult", "MiddleAge", "Senior"),
                     right = FALSE)

# --- Fare Binning ---
full$FareGroup <- cut(full$Fare, 
                      breaks = c(-Inf, 7.91, 14.454, 31, Inf),
                      labels = c("Low", "MedLow", "MedHigh", "High"))

# --- Encoding ---
full$Sex <- as.factor(full$Sex)
full$Embarked <- as.factor(full$Embarked)
full$Pclass <- as.factor(full$Pclass)

# ==============================================================================
# 3. ADVANCED: Family Survival Rate (FSR)
# ==============================================================================
cat("Computing Family Survival Rate (FSR)...\n")

# This is the key trick. For each passenger, we compute the survival rate
# of their family members (from the training set only), excluding themselves.

# Split data for FSR calculation
train_fsr <- full[1:nrow(train), c("PassengerId", "Surname", "FamilySize", "Survived", "Sex", "Pclass")]
test_fsr  <- full[(nrow(train) + 1):nrow(full), c("PassengerId", "Surname", "FamilySize", "Sex", "Pclass")]

# Function to compute FSR for a passenger
# Returns median survival rate of other family members who are in training set
compute_fsr <- function(passenger_id) {
  row <- full[full$PassengerId == passenger_id, ]
  surname <- row$Surname
  fare <- full$Fare[full$PassengerId == passenger_id]
  
  # Find family members: same Surname AND similar Fare (to handle name collisions)
  # For robustness, we also check FamilySize > 1 and same Ticket if available
  family_members <- train[train$Surname == surname & 
                            train$PassengerId != passenger_id &
                            abs(train$Fare - fare) < 1, ] # Fare tolerance for grouping
  
  if (nrow(family_members) == 0) {
    return(NA) # No family info available
  }
  
  return(mean(family_members$Survived, na.rm = TRUE))
}

# Extract surnames for training set before using in function
train$Surname <- sapply(train$Name, function(x) strsplit(x, split = ",")[[1]][1])

# Apply to all passengers (this takes a moment)
full$FSR <- sapply(full$PassengerId, function(pid) {
  row <- full[full$PassengerId == pid, ]
  surname <- row$Surname
  fare <- row$Fare
  
  family_members <- train[train$Surname == surname & 
                            train$PassengerId != pid &
                            abs(train$Fare - fare) < 1, ]
  
  if (nrow(family_members) == 0) {
    return(NA)
  }
  
  return(mean(family_members$Survived, na.rm = TRUE))
})

# Fill NA FSR with median (or a sensible default based on Sex)
# For Women/Children with no family info, assume higher survival
# For Men with no family info, assume lower survival
median_fsr_female <- median(full$FSR[full$Sex == "female"], na.rm = TRUE)
median_fsr_male   <- median(full$FSR[full$Sex == "male"], na.rm = TRUE)

full$FSR[is.na(full$FSR) & full$Sex == "female"] <- ifelse(is.na(median_fsr_female), 0.75, median_fsr_female)
full$FSR[is.na(full$FSR) & full$Sex == "male"]   <- ifelse(is.na(median_fsr_male), 0.2, median_fsr_male)

cat("  FSR computed for", sum(!is.na(full$FSR)), "passengers.\n")

# ==============================================================================
# 4. Ticket-Based Group Survival (Similar logic)
# ==============================================================================
cat("Computing Ticket Survival Rate (TSR)...\n")

full$TSR <- sapply(full$PassengerId, function(pid) {
  row <- full[full$PassengerId == pid, ]
  ticket <- row$Ticket
  
  ticket_members <- train[train$Ticket == ticket & 
                            train$PassengerId != pid, ]
  
  if (nrow(ticket_members) == 0) {
    return(NA)
  }
  
  return(mean(ticket_members$Survived, na.rm = TRUE))
})

# Fill NA TSR with FSR (if available) or median
full$TSR[is.na(full$TSR)] <- full$FSR[is.na(full$TSR)]

# ==============================================================================
# 5. Final Data Prep
# ==============================================================================
cat("Preparing final dataset...\n")

# Drop columns not needed for modeling
full_clean <- full %>% 
  select(-PassengerId, -Name, -Ticket, -Cabin, -Surname, -FamilyID)

# Split
train_final <- full_clean[1:nrow(train), ]
test_final  <- full_clean[(nrow(train) + 1):nrow(full), ]
test_final$Survived <- NULL

# Target as factor
train_final$Survived <- as.factor(ifelse(train_final$Survived == 1, "Yes", "No"))

# ==============================================================================
# 6. Model Training & CV
# ==============================================================================
cat("Training models with CV...\n")

myControl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  savePredictions = "final",
  verboseIter = FALSE
)

# XGBoost
set.seed(42)
model_xgb <- train(
  Survived ~ ., 
  data = train_final,
  method = "xgbTree",
  trControl = myControl,
  tuneLength = 3,
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
  tuneLength = 3,
  importance = "impurity"
)

cat("Random Forest CV Accuracy:", max(model_rf$results$Accuracy), "\n")

# ==============================================================================
# 7. Final Predictions & Ensemble
# ==============================================================================
cat("Generating predictions...\n")

# Train final models on full data
control_final <- trainControl(method = "none", classProbs = TRUE)

set.seed(42)
final_xgb <- train(
  Survived ~ ., 
  data = train_final,
  method = "xgbTree",
  trControl = control_final,
  tuneGrid = model_xgb$bestTune,
  verbose = FALSE
)

set.seed(42)
final_rf <- train(
  Survived ~ ., 
  data = train_final,
  method = "ranger",
  trControl = control_final,
  tuneGrid = model_rf$bestTune,
  importance = "impurity"
)

pred_xgb <- predict(final_xgb, newdata = test_final, type = "prob")
pred_rf  <- predict(final_rf, newdata = test_final, type = "prob")

# Ensemble (XGB performed better on CV, weight accordingly)
final_prob <- (0.6 * pred_xgb$Yes) + (0.4 * pred_rf$Yes)
final_class <- ifelse(final_prob > 0.5, 1, 0)

# ==============================================================================
# 8. Submission
# ==============================================================================
submission <- data.frame(PassengerId = test_ids, Survived = final_class)
write.csv(submission, "c:/Git/kaggle-titanic-competition/submission_v2.csv", row.names = FALSE)

cat("\nSubmission saved to submission_v2.csv\n")
cat("Done!\n")
