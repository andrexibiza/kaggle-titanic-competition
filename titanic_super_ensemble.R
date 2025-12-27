# ==============================================================================
# TITANIC SUPER ENSEMBLE
# Combining XGBoost (v5), Random Forest (v5), and Deep Learning (v6/Keras)
# ==============================================================================

library(caret)
library(dplyr)
library(stringr)
library(xgboost)
library(ranger)
library(keras)
library(tensorflow)

# Reproducibility
set.seed(42)
tensorflow::set_random_seed(42)

# ==============================================================================
# 1. DATA LOADING
# ==============================================================================
cat("Loading data...\n")
train <- read.csv("c:/Git/kaggle-titanic-competition/train.csv", stringsAsFactors = FALSE)
test  <- read.csv("c:/Git/kaggle-titanic-competition/test.csv",  stringsAsFactors = FALSE)
test_ids <- test$PassengerId
test$Survived <- NA
full <- bind_rows(train, test)

# ==============================================================================
# 2. FEATURE ENGINEERING (Best of v5)
# ==============================================================================
cat("Engineering features...\n")

# Title
full$Title <- str_extract(full$Name, "[a-zA-Z]+\\.")
full$Title[full$Title %in% c("Mme.")] <- "Mrs."
full$Title[full$Title %in% c("Mlle.", "Ms.")] <- "Miss."
full$Title[full$Title %in% c("Lady.", "Countess.", "Dona.")] <- "Mrs."
full$Title[full$Title %in% c("Capt.", "Col.", "Don.", "Dr.", "Major.", "Rev.", "Sir.", "Jonkheer.")] <- "Rare"
full$Title <- as.factor(full$Title)

# Ticket Group Size & FarePerPerson
ticket_counts <- table(full$Ticket)
full$TicketGroupSize <- as.integer(ticket_counts[full$Ticket])

full$Fare[is.na(full$Fare)] <- median(full$Fare, na.rm = TRUE)
full$FarePerPerson <- full$Fare / full$TicketGroupSize
full$FarePerPerson[is.infinite(full$FarePerPerson) | is.na(full$FarePerPerson)] <- median(full$FarePerPerson[is.finite(full$FarePerPerson)], na.rm = TRUE)

# Log Transform
full$Fare <- log(full$Fare + 1)
full$FarePerPerson <- log(full$FarePerPerson + 1)

# Deck
full$Deck <- ifelse(full$Cabin == "", "U", substr(full$Cabin, 1, 1))
full$Deck <- as.factor(full$Deck)

# Imputation
full$Embarked[full$Embarked == ""] <- "S"
title_age_medians <- tapply(full$Age, full$Title, median, na.rm = TRUE)
full$Age <- ifelse(is.na(full$Age), title_age_medians[full$Title], full$Age)

# Other
full$Surname <- sapply(full$Name, function(x) strsplit(x, split = ",")[[1]][1])
full$FamilySize <- full$SibSp + full$Parch + 1
full$IsAlone <- as.factor(ifelse(full$FamilySize == 1, 1, 0))

# Encoding
full$Sex <- as.factor(full$Sex)
full$Embarked <- as.factor(full$Embarked)
full$Pclass <- as.factor(full$Pclass)

# ==============================================================================
# 3. GROUP SURVIVAL (Leakage-free)
# ==============================================================================
cat("Computing group survival...\n")
# Must calculate FSR/TSR using TRAIN data only for target info
full$FSR <- sapply(1:nrow(full), function(i) {
  pid <- full$PassengerId[i]
  surname <- full$Surname[i]
  fare <- full$Fare[i]
  family <- train[train$Surname == surname & 
                    train$PassengerId != pid & 
                    abs(log(train$Fare+1) - fare) < 0.5, ]
  if (nrow(family) == 0) return(0.5) 
  mean(family$Survived, na.rm = TRUE)
})

full$TSR <- sapply(1:nrow(full), function(i) {
  pid <- full$PassengerId[i]
  ticket <- full$Ticket[i]
  group <- train[train$Ticket == ticket & train$PassengerId != pid, ]
  if (nrow(group) == 0) return(0.5)
  mean(group$Survived, na.rm = TRUE)
})

full$GroupSurvived <- pmax(full$FSR, full$TSR)

# ==============================================================================
# 4. PREP FOR MODELS
# ==============================================================================
# Drop non-model cols
full_clean <- full %>% select(-PassengerId, -Name, -Ticket, -Cabin, -Surname, -FSR, -TSR)

# -- Tree Data (Factors ok) --
train_tree <- full_clean[1:nrow(train), ]
test_tree  <- full_clean[(nrow(train) + 1):nrow(full), ]
test_tree$Survived <- NULL
train_target_factor <- as.factor(ifelse(train$Survived == 1, "Yes", "No"))

# -- Keras Data (One-Hot + Scaled) --
dummies <- dummyVars(Survived ~ ., data = full_clean)
data_x <- predict(dummies, newdata = full_clean)
preProc <- preProcess(data_x, method = c("center", "scale"))
data_x_scaled <- predict(preProc, data_x)

train_k_x <- as.matrix(data_x_scaled[1:nrow(train), ])
test_k_x  <- as.matrix(data_x_scaled[(nrow(train) + 1):nrow(full), ])
train_k_y <- as.array(as.numeric(train$Survived))

# ==============================================================================
# 5. MODEL TRAINING
# ==============================================================================
cat("Training Models...\n")

# --- XGBoost (v5 Params) ---
cat(" -> XGBoost\n")
grid_xgb <- expand.grid(
  nrounds = 100, max_depth = 4, eta = 0.1, gamma = 0, 
  colsample_bytree = 0.8, min_child_weight = 1, subsample = 0.8
)
ctrl_none <- trainControl(method = "none", classProbs = TRUE)

# caret xgbTree handles factors internally usually, but let's be safe and use matrix
# XGBoost needs fully numeric input
# Create full dummy vars for predictors ONLY
train_predictors <- train_tree %>% select(-Survived)
dummy_model <- dummyVars(~ ., data = train_predictors)
xgb_data_x <- predict(dummy_model, newdata = train_predictors)
xgb_data_y <- train_target_factor

set.seed(42)
model_xgb <- caret::train(
  x = xgb_data_x, y = xgb_data_y,
  method = "xgbTree", trControl = ctrl_none, tuneGrid = grid_xgb
)

# --- Random Forest (v5 Params) ---
cat(" -> Random Forest\n")
grid_rf <- expand.grid(mtry = 4, splitrule = "gini", min.node.size = 3)

# Ranger handles factors natively
set.seed(42)
model_rf <- caret::train(
  x = train_tree %>% select(-Survived), y = train_target_factor,
  method = "ranger", trControl = ctrl_none, tuneGrid = grid_rf, importance = "impurity"
)

# --- Deep Learning (Keras MLP) ---
cat(" -> Keras MLP\n")
input_dim <- as.integer(ncol(train_k_x))

build_model <- function(dim) {
  inputs <- layer_input(shape = c(dim))
  
  outputs <- inputs %>%
    layer_batch_normalization() %>%
    layer_dense(units = 32L, activation = 'relu') %>%
    layer_dropout(rate = 0.4) %>%
    layer_batch_normalization() %>%
    layer_dense(units = 32L, activation = 'relu') %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 1L, activation = 'sigmoid')
  
  k_model <- keras_model(inputs = inputs, outputs = outputs)
  
  k_model$compile(
    loss = 'binary_crossentropy', 
    optimizer = optimizer_adam(learning_rate = 0.001), 
    metrics = list('accuracy')
  )
  k_model
}

# Train 3 Keras models and average (Mini-Ensemble for stability)
keras_preds <- rep(0, nrow(test_k_x))
n_k_models <- 3

for(i in 1:n_k_models) {
  k_mod <- build_model(input_dim)
  k_mod$fit(
    train_k_x, train_k_y, 
    epochs = 60L, batch_size = 32L, verbose = 0L 
    # No early stopping here, just fixed epochs as we know it converges roughly here from v6
  )
  p <- k_mod$predict(test_k_x)
  keras_preds <- keras_preds + p
}
keras_preds <- keras_preds / n_k_models

# ==============================================================================
# 6. ENSEMBLE PREDICTIONS
# ==============================================================================
cat("Predicting & Blending...\n")

xgb_test_x <- predict(dummy_model, newdata = test_tree)

prob_xgb <- predict(model_xgb, newdata = xgb_test_x, type = "prob")$Yes
prob_rf  <- predict(model_rf, newdata = test_tree, type = "prob")$Yes
prob_dl  <- as.numeric(keras_preds)

# Soft Voting Weights
# XGB: 0.45, RF: 0.45, DL: 0.10
final_prob <- (0.45 * prob_xgb) + (0.45 * prob_rf) + (0.10 * prob_dl)

final_class <- ifelse(final_prob > 0.5, 1, 0)

submission <- data.frame(PassengerId = test_ids, Survived = final_class)
write.csv(submission, "c:/Git/kaggle-titanic-competition/submission_super_ensemble.csv", row.names = FALSE)

cat("Submission saved to submission_super_ensemble.csv\n")
cat("Done!\n")
