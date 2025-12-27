# ==============================================================================
# TITANIC ADVANCED MODEL v3
# Goal: Push from 0.77 to >0.80 using refined FSR and rule-based overrides
# ==============================================================================

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
cat("Engineering features...\n")

# Title
full$Title <- str_extract(full$Name, "[a-zA-Z]+\\.")
full$Title[full$Title %in% c("Mme.")] <- "Mrs."
full$Title[full$Title %in% c("Mlle.", "Ms.")] <- "Miss."
full$Title[full$Title == "Master."] <- "Master" # Keep boys separate!
full$Title[full$Title %in% c("Mrs.", "Miss.")] <- "Female"
full$Title[full$Title %in% c("Mr.")] <- "Mr"
full$Title[!full$Title %in% c("Master", "Female", "Mr")] <- "Rare"
full$Title <- as.factor(full$Title)

# Surname
full$Surname <- sapply(full$Name, function(x) strsplit(x, split = ",")[[1]][1])

# Family Size
full$FamilySize <- full$SibSp + full$Parch + 1

# Mother indicator (female with Parch > 0 and Title = Mrs./Miss, age > 18)
full$IsMother <- ifelse(full$Sex == "female" & full$Parch > 0 & full$Age > 18 & !is.na(full$Age), 1, 0)

# Child indicator
full$IsChild <- ifelse(full$Age < 16 & !is.na(full$Age), 1, 0)

# Deck
full$Deck <- ifelse(full$Cabin == "", "U", substr(full$Cabin, 1, 1))
full$Deck <- as.factor(full$Deck)

# Imputation
full$Embarked[full$Embarked == ""] <- "S"
full$Fare[is.na(full$Fare)] <- median(full$Fare, na.rm = TRUE)

# Age imputation by Title
title_age_medians <- tapply(full$Age, full$Title, median, na.rm = TRUE)
full$Age <- ifelse(is.na(full$Age), title_age_medians[full$Title], full$Age)

# Re-compute IsChild after imputation
full$IsChild <- ifelse(full$Age < 16, 1, 0)

# Age Group
full$AgeGroup <- cut(full$Age, breaks = c(0, 5, 12, 18, 35, 60, Inf), 
                     labels = c("Infant", "Child", "Teen", "Adult", "MiddleAge", "Senior"), right = FALSE)

# Fare Group (quartiles)
full$FareGroup <- cut(full$Fare, breaks = c(-Inf, 7.91, 14.454, 31, Inf),
                      labels = c("Low", "MedLow", "MedHigh", "High"))

# Encoding
full$Sex <- as.factor(full$Sex)
full$Embarked <- as.factor(full$Embarked)
full$Pclass <- as.factor(full$Pclass)

# ==============================================================================
# 3. REFINED: Family & Ticket Group Survival Rate
# ==============================================================================
cat("Computing refined group survival features...\n")

# Extract surnames for training set
train$Surname <- sapply(train$Name, function(x) strsplit(x, split = ",")[[1]][1])

# Family Survival Rate (FSR): Focus on women/children pattern
# If any FEMALE or CHILD in family survived, it's a strong signal for others
full$FamilySurvived <- sapply(1:nrow(full), function(i) {
  pid <- full$PassengerId[i]
  surname <- full$Surname[i]
  fare <- full$Fare[i]
  
  # Find family in training set
  family <- train[train$Surname == surname & 
                    train$PassengerId != pid &
                    abs(train$Fare - fare) < 5, ] # Looser tolerance
  
  if (nrow(family) == 0) return(0.5) # Unknown
  
  mean(family$Survived, na.rm = TRUE)
})

# Ticket Group Survival
full$TicketSurvived <- sapply(1:nrow(full), function(i) {
  pid <- full$PassengerId[i]
  ticket <- full$Ticket[i]
  
  group <- train[train$Ticket == ticket & train$PassengerId != pid, ]
  
  if (nrow(group) == 0) return(0.5) # Unknown
  
  mean(group$Survived, na.rm = TRUE)
})

# Combine: Use max (if either family or ticket group survived, high chance)
full$GroupSurvived <- pmax(full$FamilySurvived, full$TicketSurvived)

# ==============================================================================
# 4. Final Data Prep
# ==============================================================================
cat("Preparing final dataset...\n")

full_clean <- full %>% 
  select(-PassengerId, -Name, -Ticket, -Cabin, -Surname)

train_final <- full_clean[1:nrow(train), ]
test_final  <- full_clean[(nrow(train) + 1):nrow(full), ]
test_final$Survived <- NULL

train_final$Survived <- as.factor(ifelse(train_final$Survived == 1, "Yes", "No"))

# ==============================================================================
# 5. Hyperparameter Tuned XGBoost
# ==============================================================================
cat("Training XGBoost with hyperparameter tuning...\n")

# Expand tuning grid
xgb_grid <- expand.grid(
  nrounds = c(100, 150, 200),
  max_depth = c(3, 4, 5),
  eta = c(0.05, 0.1, 0.3),
  gamma = 0,
  colsample_bytree = c(0.7, 0.8),
  min_child_weight = c(1, 3),
  subsample = c(0.8, 1)
)

myControl <- trainControl(
  method = "cv",
  number = 5, # Faster CV
  classProbs = TRUE,
  verboseIter = FALSE
)

set.seed(42)
model_xgb <- train(
  Survived ~ ., 
  data = train_final,
  method = "xgbTree",
  trControl = myControl,
  tuneGrid = xgb_grid,
  verbose = FALSE
)

cat("Best XGBoost CV Accuracy:", max(model_xgb$results$Accuracy), "\n")
cat("Best Params:", paste(names(model_xgb$bestTune), model_xgb$bestTune, sep="=", collapse=", "), "\n")

# ==============================================================================
# 6. Final Model & Predictions
# ==============================================================================
cat("Generating predictions...\n")

control_final <- trainControl(method = "none", classProbs = TRUE)

set.seed(42)
final_model <- train(
  Survived ~ ., 
  data = train_final,
  method = "xgbTree",
  trControl = control_final,
  tuneGrid = model_xgb$bestTune,
  verbose = FALSE
)

pred_prob <- predict(final_model, newdata = test_final, type = "prob")
pred_class <- ifelse(pred_prob$Yes > 0.5, 1, 0)

# ==============================================================================
# 7. RULE-BASED OVERRIDES
# ==============================================================================
cat("Applying rule-based overrides...\n")

# Reconstruct relevant columns for test set
test_meta <- full[(nrow(train) + 1):nrow(full), c("PassengerId", "Sex", "Pclass", "Age", 
                                                     "FamilySize", "GroupSurvived", "Title")]

# Rule 1: 1st/2nd class adult females with no group info -> survive
# (About 95% survival rate historically)
rule1 <- test_meta$Sex == "female" & 
         test_meta$Pclass %in% c(1, 2) & 
         test_meta$GroupSurvived >= 0.5

# Rule 2: 3rd class adult males alone -> die
# (About 10% survival rate)
rule2 <- test_meta$Sex == "male" & 
         test_meta$Pclass == 3 & 
         test_meta$FamilySize == 1 &
         test_meta$GroupSurvived <= 0.5

# Rule 3: Children under 10 with GroupSurvived > 0 -> survive
rule3 <- test_meta$Age < 10 & test_meta$GroupSurvived > 0.5

# Rule 4: Women with GroupSurvived == 0 (family died) -> die
rule4 <- test_meta$Sex == "female" & test_meta$GroupSurvived == 0

# Apply overrides
pred_class[rule1] <- 1
pred_class[rule2] <- 0
pred_class[rule3] <- 1
pred_class[rule4] <- 0

# ==============================================================================
# 8. Submission
# ==============================================================================
submission <- data.frame(PassengerId = test_ids, Survived = pred_class)
write.csv(submission, "c:/Git/kaggle-titanic-competition/submission_v3.csv", row.names = FALSE)

cat("\nSubmission saved to submission_v3.csv\n")
cat("Done!\n")
