
# Load required libraries
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(ranger)) install.packages("ranger", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")

library(caret)
library(dplyr)
library(stringr)
library(kernlab)

# ==============================================================================
# 1. Data Loading & Preparation
# ==============================================================================
cat("Loading data...\n")
train <- read.csv("c:/Git/kaggle-titanic-competition/train.csv", stringsAsFactors = FALSE)
test  <- read.csv("c:/Git/kaggle-titanic-competition/test.csv",  stringsAsFactors = FALSE)

test$Survived <- NA
full <- bind_rows(train, test)

# ==============================================================================
# 2. Advanced Feature Engineering
# ==============================================================================
cat("Engineering features...\n")

# --- Title Extraction & Consolidation ---
full$Title <- str_extract(full$Name, "[a-zA-Z]+\\.")
# Map rare titles
full$Title[full$Title %in% c("Mme.", "Mlle.")] <- "Miss."
full$Title[full$Title %in% c("Lady.", "the Countess.", "Capt.", "Col.", "Don.", "Dr.", "Major.", "Rev.", "Sir.", "Jonkheer.", "Dona.")] <- "Rare"
full$Title[full$Title == "Ms."] <- "Miss." # or Mrs, usually Miss
full$Title <- as.factor(full$Title)

# --- Family Size ---
full$FamilySize <- full$SibSp + full$Parch + 1
full$IsAlone <- ifelse(full$FamilySize == 1, 1, 0)

# --- Deck Extraction ---
# If Cabin is missing, set to "U" (Unknown)
full$Deck <- ifelse(full$Cabin == "", "U", substr(full$Cabin, 1, 1))
full$Deck <- as.factor(full$Deck)

# --- Imputation (Simple for Benchmark) ---
# Age: Median by Title (better than global median)
# Fare: Median (only 1 missing in test)
# Embarked: Mode
full$Embarked[full$Embarked == ""] <- "S"
full$Fare[is.na(full$Fare)] <- median(full$Fare, na.rm = TRUE)

title_age_medians <- tapply(full$Age, full$Title, median, na.rm = TRUE)
# Function to fill NA Age based on Title
fill_age <- function(age, title) {
  if (is.na(age)) {
    return(title_age_medians[title])
  } else {
    return(age)
  }
}
full$Age <- mapply(fill_age, full$Age, full$Title)

# --- Encoding & Scaling ---
full$Sex <- as.factor(full$Sex)
full$Embarked <- as.factor(full$Embarked)
full$Pclass <- as.factor(full$Pclass) # Treat as factor for some models, or ordered

# Drop High Cardinality / Unused
full <- full %>% select(-PassengerId, -Name, -Ticket, -Cabin)

# Split back
train_clean <- full[1:nrow(train), ]
test_clean  <- full[(nrow(train) + 1):nrow(full), ]
test_clean$Survived <- NULL

# Ensure target is a factor for Classification
train_clean$Survived <- as.factor(ifelse(train_clean$Survived == 1, "Yes", "No"))

# ==============================================================================
# 3. Model Benchmarking
# ==============================================================================
cat("Training models (10-fold CV)...\n")

# Shared Control for fair comparison
myControl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE, # Needed for AUC if we wanted it
  savePredictions = "final",
  verboseIter = FALSE
)

# --- Model 1: Random Forest (ranger) ---
cat("Training Random Forest...\n")
set.seed(42)
model_rf <- train(
  Survived ~ ., 
  data = train_clean,
  method = "ranger",
  trControl = myControl,
  tuneLength = 5,
  importance = "impurity"
)

# --- Model 2: XGBoost ---
cat("Training XGBoost...\n")
set.seed(42)
# XGBoost requires numeric matrix usually, but caret handles factors internally often.
# However, explicit dummy vars are safer for XGB.
# For simplicity in this script, we trust caret's internal handling or let it fail/adjust.
model_xgb <- train(
  Survived ~ ., 
  data = train_clean,
  method = "xgbTree",
  trControl = myControl,
  tuneLength = 3,
  verbose = FALSE
)

# --- Model 3: SVM Radial ---
cat("Training SVM Radial...\n")
set.seed(42)
model_svm <- train(
  Survived ~ ., 
  data = train_clean,
  method = "svmRadial",
  trControl = myControl,
  preProcess = c("center", "scale"), # SVM needs scaling
  tuneLength = 5
)

# --- Model 4: GLMnet (Elastic Net) ---
cat("Training GLMnet...\n")
set.seed(42)
model_glm <- train(
  Survived ~ ., 
  data = train_clean,
  method = "glmnet",
  trControl = myControl,
  preProcess = c("center", "scale"),
  tuneLength = 5
)

# ==============================================================================
# 4. Results Comparison
# ==============================================================================
results <- resamples(list(
  RF = model_rf,
  XGB = model_xgb,
  SVM = model_svm,
  GLM = model_glm
))

cat("\n--- Model Comparison Results ---\n")
summary(results)

cat("\n--- Correlations between models ---\n")
modelCor(results)

# Pick best model (based on highest mean Accuracy)
accuracies <- results$values %>% select(ends_with("Accuracy"))
mean_accuracies <- colMeans(accuracies, na.rm = TRUE)
best_model_name <- names(sort(mean_accuracies, decreasing = TRUE))[1]

cat("\nBest performing model based on CV Accuracy:", best_model_name, "\n")
