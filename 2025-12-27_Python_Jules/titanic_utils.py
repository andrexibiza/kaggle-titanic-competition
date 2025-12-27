import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin

def load_data():
    """Loads train and test data."""
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    return train, test

def extract_title(name):
    title = re.search(r"([A-Za-z]+)\.", name)
    if title:
        return title.group(1)
    return ""

def clean_title(title):
    if title in ["Mme"]: return "Mrs"
    if title in ["Mlle", "Ms"]: return "Miss"
    if title in ["Lady", "Countess", "Dona"]: return "Mrs"
    if title in ["Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer"]: return "Rare"
    return title

class V4FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.title_age_medians = {}
        self.fare_median = 0
        self.embarked_mode = 'S'
        self.train_data = None # Store training data for target encoding

    def fit(self, X, y=None):
        # We need to store the training set (or the set passed to fit)
        # to calculate target encodings later if we want to be strict,
        # but typically we just learn the medians here.

        # NOTE: For Family/Ticket Survival, V4.R uses the WHOLE dataset (Train+Test)
        # to find families, but only uses TRAIN data to calculate survival rates.
        # This is tricky to fit into a standard sklearn transformer if we process Train/Test separately.
        # However, for this competition, it's standard to combine them first.

        # Learn Medians for Age imputation
        temp_df = X.copy()
        temp_df['Title'] = temp_df['Name'].apply(extract_title).apply(clean_title)
        self.title_age_medians = temp_df.groupby('Title')['Age'].median().to_dict()

        self.fare_median = temp_df['Fare'].median()
        self.embarked_mode = temp_df['Embarked'].mode()[0]

        return self

    def transform(self, X):
        df = X.copy()

        # 1. Title
        df['Title'] = df['Name'].apply(extract_title).apply(clean_title)

        # 2. Imputation
        df['Embarked'] = df['Embarked'].fillna(self.embarked_mode)
        df['Fare'] = df['Fare'].fillna(self.fare_median)

        # Age Imputation
        def fill_age(row):
            if pd.isna(row['Age']):
                return self.title_age_medians.get(row['Title'], df['Age'].median())
            return row['Age']

        df['Age'] = df.apply(fill_age, axis=1)

        # 3. Deck
        df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) and x != "" else "U")

        # 4. Family Size
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        # 5. Encoding (Label Encoding for simplicity, or OneHot later)
        # For now, let's keep them as strings/categories or simple maps
        # V4 uses factors in R.

        return df

def get_family_survival_rate(full_df, train_limit_index):
    """
    Replicates the logic from V4.R:
    Calculates FamilySurvived and TicketSurvived.
    Using 'train' part of the data to calculate rates for everyone.
    """

    # Setup
    df = full_df.copy()
    df['Surname'] = df['Name'].apply(lambda x: x.split(',')[0])

    # Indices
    train_indices = range(train_limit_index)

    # We need to compute for every row in full_df
    family_survived = []
    ticket_survived = []

    # This can be slow, but for 1309 rows it's fine.
    # To match V4.R exactly:
    # "family <- train[train$Surname == surname & train$PassengerId != pid & abs(train$Fare - fare) < 5, ]"

    # Pre-filter training data to speed up
    train_subset = df.iloc[:train_limit_index].copy()

    for i in range(len(df)):
        passenger = df.iloc[i]
        pid = passenger['PassengerId']
        surname = passenger['Surname']
        fare = passenger['Fare']
        ticket = passenger['Ticket']

        # Family Survival
        # Find matches in TRAINING set (excluding self if self is in training)
        if i < train_limit_index:
             # If I am in training set, exclude myself
             family_matches = train_subset[
                 (train_subset['Surname'] == surname) &
                 (train_subset['PassengerId'] != pid) &
                 (abs(train_subset['Fare'] - fare) < 0.001) # V4.R used < 5, but typically strict equality or close is implied for "same family". R code said < 5. Let's stick to V4.R
             ]
             # Correcting: V4.R says "abs(train$Fare - fare) < 5".
             family_matches = train_subset[
                 (train_subset['Surname'] == surname) &
                 (train_subset['PassengerId'] != pid) &
                 (abs(train_subset['Fare'] - fare) < 5)
             ]
        else:
             # If I am in test set, look at all training set
             family_matches = train_subset[
                 (train_subset['Surname'] == surname) &
                 (abs(train_subset['Fare'] - fare) < 5)
             ]

        if len(family_matches) > 0:
            family_survived.append(family_matches['Survived'].mean())
        else:
            family_survived.append(0.5)

        # Ticket Survival
        if i < train_limit_index:
            ticket_matches = train_subset[
                (train_subset['Ticket'] == ticket) &
                (train_subset['PassengerId'] != pid)
            ]
        else:
            ticket_matches = train_subset[
                (train_subset['Ticket'] == ticket)
            ]

        if len(ticket_matches) > 0:
            ticket_survived.append(ticket_matches['Survived'].mean())
        else:
            ticket_survived.append(0.5)

    df['FamilySurvived'] = family_survived
    df['TicketSurvived'] = ticket_survived
    df['GroupSurvived'] = df[['FamilySurvived', 'TicketSurvived']].max(axis=1)

    return df[['FamilySurvived', 'TicketSurvived', 'GroupSurvived']]

def prepare_v4_data():
    train, test = load_data()
    train_len = len(train)

    # Combine
    full = pd.concat([train.drop('Survived', axis=1), test], axis=0, ignore_index=True)

    # Basic FE
    fe = V4FeatureEngineer()
    fe.fit(full) # Fit on full to get global medians? Or just train?
                 # V4 R: "Fare[is.na(Fare)] <- median(full$Fare)". It uses full.
    full = fe.transform(full)

    # Advanced FE (Target Encoding)
    # We need the Survived column back in the training part of 'full' for the function to work
    full_with_target = full.copy()
    full_with_target.loc[:train_len-1, 'Survived'] = train['Survived'].values

    group_feats = get_family_survival_rate(full_with_target, train_len)
    full = pd.concat([full, group_feats], axis=1)

    # Prepare final frames
    # Select columns as per V4
    # "full_clean <- full %>% select(-PassengerId, -Name, -Ticket, -Cabin, -Surname)"
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    full_clean = full.drop(cols_to_drop, axis=1)

    # Encoding for categorical
    # R uses factors. XGBoost/CatBoost can handle categories, but sklearn needs numbers.
    # Let's One-Hot Encode 'Sex', 'Embarked', 'Title', 'Deck'
    full_clean = pd.get_dummies(full_clean, columns=['Sex', 'Embarked', 'Title', 'Deck'], drop_first=True)

    X_train = full_clean.iloc[:train_len].copy()
    y_train = train['Survived']
    X_test = full_clean.iloc[train_len:].copy()

    return X_train, y_train, X_test, test['PassengerId']
