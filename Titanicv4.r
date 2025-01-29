# Load packages
library(caret)        # machine learning
library(dplyr)        # data manipulation
library(ggplot2)      # viz
library(Hmisc)        # robust describe() function
library(naniar)       # working with missing data
library(randomForest) # inference model

# Load train and test data
train <- read.csv("train.csv", stringsAsFactors = FALSE)
test <- read.csv("test.csv", stringsAsFactors = FALSE)
head(train) #--loaded successfully
head(test)  #--loaded successfully

# Evaluate structure and data types
# str(train)
# str(test)
# 
# describe(train)
# train has missing values: Age 177, Cabin 687, Embarked 2
# describe(test)
# test has missing values: Cabin 327, Fare 1, Age 86

# DATA CLEANING AND PREPROCESSING
# 1) Encode categorical variables
# [X] Encode Sex as numeric factor
train$Sex <- as.factor(ifelse(train$Sex == "male", 1, 0)) # v2.2 added as.factor() to coerce output
test$Sex <- as.factor(ifelse(test$Sex == "male", 1, 0))
head(train[, "Sex"]) #--encoded successfully
head(test[, "Sex"]) #--encoded successfully

# [X] Convert Pclass to an ordinal factor
train$Pclass <- factor(train$Pclass, levels = c(1, 2, 3), ordered = TRUE)
test$Pclass <- factor(test$Pclass, levels = c(1, 2, 3), ordered = TRUE)
head(train[, "Pclass"]) #--encoded successfully
head(test[, "Pclass"]) #--encoded successfully

# [X] One-hot encode Embarked
embarked_train_one_hot <- model.matrix(~ Embarked - 1, data = train)
embarked_test_one_hot <- model.matrix(~ Embarked - 1, data = test)

# Add the one-hot encoded columns back to the dataset
train <- cbind(train, embarked_train_one_hot)
test <- cbind(test, embarked_test_one_hot)

# Verify encoding:
head(train[, c("Embarked", "EmbarkedC", "EmbarkedQ", "EmbarkedS")])
head(test[, c("Embarked", "EmbarkedC", "EmbarkedQ", "EmbarkedS")])

# -- looks perfect, let's not forget about imputing our 2 missing values
# Impute 2 missing Embarked values with the mode
train$Embarked[train$Embarked == ""] <- NA
# Calculate the mode and ensure it's a single value
embarked_mode <- names(sort(table(train$Embarked), decreasing = TRUE))[1]
train$Embarked[is.na(train$Embarked)] <- embarked_mode

# verify imputation
describe(train$Embarked)

##v2.2 also want to explicitly cast the values in EmbarkedC, EmbarkedQ, and EmbarkedS as factors.
train$EmbarkedC <- as.factor(train$EmbarkedC)
test$EmbarkedC <- as.factor(test$EmbarkedC)
train$EmbarkedQ <- as.factor(train$EmbarkedQ)
test$EmbarkedQ <- as.factor(test$EmbarkedQ)
train$EmbarkedS <- as.factor(train$EmbarkedS)
test$EmbarkedS <- as.factor(test$EmbarkedS)

## SibSp and Parch should be integers
train$SibSp <- as.integer(train$SibSp)
test$SibSp <- as.integer(test$SibSp)
train$Parch <- as.integer(train$Parch)
test$Parch <- as.integer(test$Parch)

# Survived needs to be a factor
train$Survived <- as.factor(train$Survived)

# 3) Address missing values
# Age - Train
#--Predict missing ages using other features
train_age_data <- train %>% 
    select(Age, Pclass, Sex, SibSp, Parch, Fare, EmbarkedC, EmbarkedQ, EmbarkedS)

# head(train[, c("Age", "Pclass", "Sex", "SibSp", "Parch", "Fare", "EmbarkedC", "EmbarkedQ", "EmbarkedS")])
#--verified that all these columns are formatted properly

train_age_complete <- train_age_data %>% filter(!is.na(Age))
train_age_missing <- train_age_data %>% filter(is.na(Age))

set.seed(666)
cv_control <- trainControl(method = "cv", number = 10) #v2.2 10-fold cross-validation for imputing missing ages
train_age_cv_model <- train(
  Age ~ Pclass + Sex + SibSp + Parch + Fare + EmbarkedC + EmbarkedQ + EmbarkedS,
  data = train_age_complete,
  method = "rf",
  trControl = cv_control,
  tuneLength = 3
)
print(train_age_cv_model)

# Use the best model to predict missing ages
predicted_train_ages <- predict(train_age_cv_model, newdata = train_age_missing)

# Impute the predicted ages back into the train dataset
train$Age[is.na(train$Age)] <- predicted_train_ages
describe(train$Age)

#--Age in test data
# Preprocess the test data for Age imputation
test_age_data <- test %>% 
  select(Age, Pclass, Sex, SibSp, Parch, Fare, EmbarkedC, EmbarkedQ, EmbarkedS)

test_age_missing <- test_age_data %>% filter(is.na(Age))
test_age_complete <- test_age_data %>% filter(!is.na(Age))

# Use the trained train_age_cv_model to predict missing ages in the test dataset
predicted_test_ages <- predict(train_age_cv_model, newdata = test_age_missing)

# Impute the predicted ages back into the test dataset
test$Age[is.na(test$Age)] <- predicted_test_ages

n_miss(test$Age)

library(stringr)

## Feature Engineering - transform Name into Title
# Update the regex pattern to include all titles
title_pattern <- "Mr|Mrs|Miss|Master|Don|Rev|Dr|Mme|Ms|Major|Lady|Sir|Mlle|Col|Capt|Countess|Jonkheer"

# Extract titles using the regex title_pattern
train$Title <- as.factor(str_extract(train$Name, title_pattern))
test$Title <- as.factor(str_extract(test$Name, title_pattern))

str(train)
str(test)

# Convert empty strings to NA in Cabin
train$Cabin[train$Cabin == ""] <- NA
test$Cabin[test$Cabin == ""] <- NA

# Create new `Deck` feature
train$Deck <- as.factor(ifelse(!is.na(train$Cabin), substr(train$Cabin, 1, 1), NA))
test$Deck <- as.factor(ifelse(!is.na(test$Cabin), substr(test$Cabin, 1, 1), NA))

# Verify the new Deck feature
#head(train[, "Cabin"])
#head(test[, "Cabin"])

# Replace NA values in Deck with "U"
train$Deck[is.na(train$Deck)] <- "U"
test$Deck[is.na(test$Deck)] <- "U"

# Verify the new Deck feature
head(train[, "Deck"])
head(test[, "Deck"])

# Encode the HasCabin variable:
train$HasCabin <- as.factor(ifelse(!is.na(train$Cabin), 1, 0))
test$HasCabin <- as.factor(ifelse(!is.na(test$Cabin), 1, 0))

# describe(train$HasCabin) # - perfect
head(train[, c("Cabin", "HasCabin")])  #looks good
head(test[, c("Cabin", "HasCabin")]) 

n_miss(train$HasCabin)
n_miss(test$HasCabin)

# Create the FamilySize feature
train$FamilySize <- as.integer(train$SibSp + train$Parch + 1)
test$FamilySize <- as.integer(test$SibSp + test$Parch + 1)

# Inspect the new feature
head(train[, "FamilySize"])
head(test[, "FamilySize"])

# describe(train)
# describe(test)
#--test still has 1 missing fare - impute with the median
test$Fare[is.na(test$Fare)] <- median(test$Fare, na.rm = TRUE)
describe(test)

# Create the GroupSize feature based on Ticket
train$GroupSize <- train %>%
  group_by(Ticket) %>%
  mutate(GroupSize = n()) %>%
  ungroup() %>%
  pull(GroupSize)

test$GroupSize <- test %>%
  group_by(Ticket) %>%
  mutate(GroupSize = n()) %>%
  ungroup() %>%
  pull(GroupSize)

# Inspect the new feature
head(train[, c("Ticket", "GroupSize")])
head(test[, c("Ticket", "GroupSize")])

# Create the FarePerPerson feature
train$FarePerPerson <- train$Fare / train$GroupSize
test$FarePerPerson <- test$Fare / test$GroupSize

# Inspect the new feature
head(train[, c("Ticket", "Fare", "GroupSize", "FarePerPerson")])
head(test[, c("Ticket", "Fare", "GroupSize", "FarePerPerson")])

# Apply log transformation to Fare and FarePerPerson
#--plot shape before transformation?
ggplot(train, aes(x = Fare)) +
  geom_histogram(bins=20) +
  theme_minimal() +
  ggtitle("Fare (before transforming)")

#--note an extreme outlier over 500!
train$Fare <- log(train$Fare + 1)
train$FarePerPerson <- log(train$FarePerPerson + 1)
test$Fare <- log(test$Fare + 1)
test$FarePerPerson <- log(test$FarePerPerson + 1)
head(train[, c("Fare", "FarePerPerson")])
head(test[, c("Fare", "FarePerPerson")])

# plot FarePerPerson before and after transformation
ggplot(train, aes(x = FarePerPerson)) +
  geom_histogram(bins=20) +
  theme_minimal() +
  ggtitle("FarePerPerson (before transforming)")

ggplot(train, aes(x = FarePerPerson)) +
  geom_histogram(bins=20) +
  theme_minimal() +
  ggtitle("Log Transformed FarePerPerson")

# Create the ChildInFamily feature
train$ChildInFamily <- as.factor(ifelse(train$Age < 15 & train$FamilySize > 1, 1, 0))
test$ChildInFamily <- as.factor(ifelse(test$Age < 15 & test$FamilySize > 1, 1, 0))

# Inspect the new feature
head(train[, c("Age", "FamilySize", "ChildInFamily")])
head(test[, c("Age", "FamilySize", "ChildInFamily")])

# Explicitly cast as integers
train$GroupSize <- as.integer(train$GroupSize)
test$GroupSize <- as.integer(test$GroupSize)
train$FamilySize <- as.integer(train$FamilySize)
test$FamilySize <- as.integer(test$FamilySize)
train$SibSp <- as.integer(train$SibSp)
test$SibSp <- as.integer(test$SibSp)
train$Parch <- as.integer(train$Parch)
test$Parch <- as.integer(test$Parch)

# drop Name, Ticket, Cabin, Embarked
train <- train %>% select(-Name, -Ticket, -Cabin, -Embarked)
test <- test %>% select(-Name, -Ticket, -Cabin, -Embarked)

str(train)
str(test)

# Use ensemble model, random forest and xgboost, to predict Survived
# Load necessary libraries
library(randomForest)
library(caret)

# Train the random forest model
set.seed(666)
rf_cv_control <- trainControl(method = "cv", number = 10)
rf_model <- train(
  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + EmbarkedC + EmbarkedQ + EmbarkedS + HasCabin + FamilySize + Title + Deck + GroupSize + FarePerPerson + ChildInFamily, 
  data = train,
  method = "rf",
  trControl = rf_cv_control,
    tuneLength = 10
)

# Print the cross-validation results
print(rf_model)

# Use the trained random forest model to predict Survived in the test dataset
test$Survived <- predict(rf_model, newdata = test)

# Create submission file
submission <- test %>% select(PassengerId, Survived)
write.csv(submission, "submission.csv", row.names = FALSE)

# Extract feature importance
importance_values <- varImp(rf_model, scale = FALSE)

# Convert to a data frame for easier plotting
importance_df <- as.data.frame(importance_values$importance)
importance_df$Feature <- rownames(importance_df)

# Plot feature importance
ggplot(importance_df, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Feature Importance from Random Forest Model",
       x = "Feature",
       y = "Importance")

# Print the feature importance values
print(importance_df)
