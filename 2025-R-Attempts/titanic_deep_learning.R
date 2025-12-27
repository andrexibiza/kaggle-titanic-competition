# ==============================================================================
# TITANIC Deep Learning v1 (Keras)
# Using v5 features + MLP with BatchCheck/Dropout
# ==============================================================================

library(caret)
library(dplyr)
library(stringr)
library(keras)
library(tensorflow)

# Set seeds for reproducibility
set.seed(42)
tensorflow::set_random_seed(42)

cat("Loading data...\n")
train <- read.csv("c:/Git/kaggle-titanic-competition/train.csv", stringsAsFactors = FALSE)
test  <- read.csv("c:/Git/kaggle-titanic-competition/test.csv",  stringsAsFactors = FALSE)

train_ids <- train$PassengerId
test_ids  <- test$PassengerId

test$Survived <- NA
full <- bind_rows(train, test)

# ==============================================================================
# Feature Engineering (From v5)
# ==============================================================================
cat("Engineering features...\n")

# Title
full$Title <- str_extract(full$Name, "[a-zA-Z]+\\.")
full$Title[full$Title %in% c("Mme.")] <- "Mrs."
full$Title[full$Title %in% c("Mlle.", "Ms.")] <- "Miss."
full$Title[full$Title %in% c("Lady.", "Countess.", "Dona.")] <- "Mrs."
full$Title[full$Title %in% c("Capt.", "Col.", "Don.", "Dr.", "Major.", "Rev.", "Sir.", "Jonkheer.")] <- "Rare"
full$Title <- as.factor(full$Title)

# Surname & Family Size
full$Surname <- sapply(full$Name, function(x) strsplit(x, split = ",")[[1]][1])
full$FamilySize <- full$SibSp + full$Parch + 1

# Ticket Group Size
ticket_counts <- table(full$Ticket)
full$TicketGroupSize <- as.integer(ticket_counts[full$Ticket])

# FarePerPerson
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

# Log Transform Skewed Numericals
full$Fare <- log(full$Fare + 1)
full$FarePerPerson <- log(full$FarePerPerson + 1)
# Age is better scaled later (MinMax or Standard)

# ==============================================================================
# Family/Ticket Survival (Leakage-free)
# ==============================================================================
cat("Computing group survival features...\n")

# Extract surnames for train
train$Surname <- sapply(train$Name, function(x) strsplit(x, split = ",")[[1]][1])

full$FSR <- sapply(1:nrow(full), function(i) {
  pid <- full$PassengerId[i]
  surname <- full$Surname[i]
  fare <- full$Fare[i]
  # Match logic: Same Surname roughly same Fare
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
# Preprocessing for Keras
# ==============================================================================
cat("Preparing data for Keras...\n")

# Select columns
# Note: Keras needs ALL numeric input. 
# We will use One-Hot Encoding for Factors and Scaling for Numericals.

keep_cols <- c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", 
               "Embarked", "Title", "Deck", "FamilySize", 
               "TicketGroupSize", "FarePerPerson", "GroupSurvived", "Survived")

data_k <- full[, keep_cols]

# Create Dummy Variables (One-Hot)
# caret::dummyVars is great for this
dummies <- dummyVars(Survived ~ ., data = data_k)
data_x <- predict(dummies, newdata = data_k)

# Split back
train_x <- data_x[1:nrow(train), ]
test_x  <- data_x[(nrow(train) + 1):nrow(full), ]
train_y <- as.numeric(train$Survived) # 0/1

# Scale Numerical Data (Z-score normalization)
# Ideally scale train, then apply to test to prevent leakage.
# But for simplicity here with caret/base:
preProc <- preProcess(train_x, method = c("center", "scale"))
train_x <- predict(preProc, train_x)
test_x  <- predict(preProc, test_x)

# Convert to Matrix
train_x <- as.matrix(train_x)
test_x  <- as.matrix(test_x)
train_y <- as.array(train_y)

cat("Input shape:", ncol(train_x), "\n")

# ==============================================================================
# Define Model
# ==============================================================================
# ==============================================================================
# Define Model
# ==============================================================================
build_model <- function(input_dim) {
  # Ensure input_dim is integer
  input_dim <- as.integer(input_dim)
  inputs <- layer_input(shape = c(input_dim))
  
  outputs <- inputs %>%
    layer_batch_normalization() %>%
    layer_dense(units = 32L, activation = 'relu') %>%
    layer_dropout(rate = 0.4) %>%
    
    layer_batch_normalization() %>%
    layer_dense(units = 32L, activation = 'relu') %>%
    layer_dropout(rate = 0.3) %>%
    
    layer_dense(units = 16L, activation = 'relu') %>%
    layer_dropout(rate = 0.2) %>%
    
    layer_dense(units = 1L, activation = 'sigmoid')
  
  model <- keras_model(inputs = inputs, outputs = outputs)
  
  model$compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer_adam(learning_rate = 0.001),
    metrics = list('accuracy')
  )
  
  return(model)
}

# ==============================================================================
# Train with K-Fold CV (Ensemble)
# ==============================================================================
cat("Training K-Fold Ensemble...\n")

k <- 5L
folds <- createFolds(train_y, k = k, list = TRUE, returnTrain = TRUE)
preds_test_list <- list()
cv_scores <- c()
input_dim <- as.integer(ncol(train_x))

for (i in 1:k) {
  cat("Fold", i, "...\n")
  
  # Split and force Matrix/Array structure
  idx_train <- folds[[i]]
  
  x_tr <- as.matrix(train_x[idx_train, ])
  y_tr <- array(as.numeric(train_y[idx_train]), dim = c(length(idx_train)))
  
  x_val <- as.matrix(train_x[-idx_train, ])
  y_val <- array(as.numeric(train_y[-idx_train]), dim = c(length(train_y[-idx_train])))
  
  # Build
  model <- build_model(input_dim)
  
  # Callbacks (Simplified)
  callbacks <- list(
    callback_early_stopping(monitor = "val_loss", patience = 15L, restore_best_weights = TRUE)
  )
  
  # Train
  history <- model$fit(
    x_tr, y_tr,
    epochs = 100L,
    batch_size = 32L,
    validation_data = list(x_val, y_val),
    callbacks = callbacks,
    verbose = 0L
  )
  
  # Evaluate
  score <- model$evaluate(x_val, y_val, verbose = 0)
  cv_scores <- c(cv_scores, score["accuracy"])
  cat("  Acc:", score["accuracy"], "\n")
  
  # Predict
  preds_test_list[[i]] <- model$predict(test_x)
}

cat("\nMean CV Accuracy:", mean(cv_scores), "\n")

# ==============================================================================
# Ensemble & Submission
# ==============================================================================
# Average predictions across folds
final_preds <- Reduce("+", preds_test_list) / k

# Threshold
final_class <- ifelse(final_preds > 0.5, 1, 0)

submission <- data.frame(PassengerId = test_ids, Survived = final_class)
write.csv(submission, "c:/Git/kaggle-titanic-competition/submission_dl_v1.csv", row.names = FALSE)

cat("Submission saved to submission_dl_v1.csv\n")
