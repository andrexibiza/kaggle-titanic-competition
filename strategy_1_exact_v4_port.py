#!/usr/bin/env python3
"""
Strategy 1: Exact V4 Port with Fare Proximity Filter
=====================================================

This is an EXACT port of the V4 R champion solution (0.78947).
Key difference from previous attempts:
- FamilySurvived includes fare proximity filter (abs(fare - fare) < 5)
- Matches R's factor encoding behavior
- Uses exact same hyperparameters

Target: Match or exceed 0.78947
"""

import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def extract_title(name):
    """Extract title from name - matching R's str_extract exactly."""
    match = re.search(r'([A-Za-z]+)\.', name)
    if match:
        return match.group(1) + '.'
    return 'Unknown.'

def map_title(title):
    """Map titles exactly as V4 R code does."""
    if title in ['Mme.']:
        return 'Mrs.'
    if title in ['Mlle.', 'Ms.']:
        return 'Miss.'
    if title in ['Lady.', 'Countess.', 'Dona.']:
        return 'Mrs.'
    if title in ['Capt.', 'Col.', 'Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.']:
        return 'Rare'
    return title

def main():
    print("=" * 60)
    print("STRATEGY 1: EXACT V4 PORT")
    print("=" * 60)

    # Load data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # Store IDs
    train_ids = train['PassengerId'].values
    test_ids = test['PassengerId'].values

    # Combine for consistent processing (like R's bind_rows)
    test['Survived'] = np.nan
    full = pd.concat([train, test], ignore_index=True)

    print(f"\nTotal samples: {len(full)}")
    print(f"Training: {len(train)}, Test: {len(test)}")

    # =========================================================================
    # FEATURE ENGINEERING (Exact V4 replication)
    # =========================================================================
    print("\n[1] Feature Engineering...")

    # Title extraction and mapping
    full['Title'] = full['Name'].apply(extract_title)
    full['Title'] = full['Title'].apply(map_title)

    # Surname extraction
    full['Surname'] = full['Name'].apply(lambda x: x.split(',')[0])

    # Family Size
    full['FamilySize'] = full['SibSp'] + full['Parch'] + 1
    full['IsAlone'] = (full['FamilySize'] == 1).astype(int)

    # Deck
    full['Deck'] = full['Cabin'].apply(lambda x: 'U' if pd.isna(x) or x == '' else x[0])

    # Imputation
    full['Embarked'] = full['Embarked'].fillna('S')
    full['Fare'] = full['Fare'].fillna(full['Fare'].median())

    # Age imputation by Title median (like R's tapply)
    title_age_medians = full.groupby('Title')['Age'].transform('median')
    full['Age'] = full['Age'].fillna(title_age_medians)
    full['Age'] = full['Age'].fillna(full['Age'].median())  # Fallback

    # Age Group (matching R's cut with right=FALSE)
    bins = [0, 12, 18, 35, 60, float('inf')]
    labels = ['Child', 'Teen', 'Adult', 'MiddleAge', 'Senior']
    full['AgeGroup'] = pd.cut(full['Age'], bins=bins, labels=labels, right=False)

    # Fare Group
    fare_bins = [-float('inf'), 7.91, 14.454, 31, float('inf')]
    fare_labels = ['Low', 'MedLow', 'MedHigh', 'High']
    full['FareGroup'] = pd.cut(full['Fare'], bins=fare_bins, labels=fare_labels)

    # =========================================================================
    # FAMILY/TICKET SURVIVAL RATE - THE KEY FEATURE!
    # =========================================================================
    print("\n[2] Computing Group Survival Features (with fare proximity)...")

    # Get training data for lookups
    train_data = full.iloc[:len(train)].copy()

    def calculate_family_survived(row_idx):
        """
        Calculate FamilySurvived exactly as V4 R code:
        - Same surname
        - Different PassengerId
        - Fare within $5 (THE KEY DIFFERENCE!)
        """
        pid = full.loc[row_idx, 'PassengerId']
        surname = full.loc[row_idx, 'Surname']
        fare = full.loc[row_idx, 'Fare']

        # Find family members in TRAINING SET ONLY
        mask = (
            (train_data['Surname'] == surname) &
            (train_data['PassengerId'] != pid) &
            (abs(train_data['Fare'] - fare) < 5)  # THE CRUCIAL FILTER
        )

        family = train_data[mask]

        if len(family) == 0:
            return 0.5  # Default - this acts as categorical flag

        return family['Survived'].mean()

    def calculate_ticket_survived(row_idx):
        """
        Calculate TicketSurvived exactly as V4 R code:
        - Same ticket
        - Different PassengerId
        """
        pid = full.loc[row_idx, 'PassengerId']
        ticket = full.loc[row_idx, 'Ticket']

        # Find ticket group in TRAINING SET ONLY
        mask = (
            (train_data['Ticket'] == ticket) &
            (train_data['PassengerId'] != pid)
        )

        group = train_data[mask]

        if len(group) == 0:
            return 0.5  # Default

        return group['Survived'].mean()

    # Calculate for all passengers
    print("    Calculating FamilySurvived...")
    full['FamilySurvived'] = [calculate_family_survived(i) for i in range(len(full))]

    print("    Calculating TicketSurvived...")
    full['TicketSurvived'] = [calculate_ticket_survived(i) for i in range(len(full))]

    # GroupSurvived = max of both
    full['GroupSurvived'] = full[['FamilySurvived', 'TicketSurvived']].max(axis=1)

    # =========================================================================
    # PREPARE FEATURES
    # =========================================================================
    print("\n[3] Preparing features...")

    # Encode categorical variables
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    le_title = LabelEncoder()
    le_deck = LabelEncoder()
    le_agegroup = LabelEncoder()
    le_faregroup = LabelEncoder()

    full['Sex_Enc'] = le_sex.fit_transform(full['Sex'])
    full['Embarked_Enc'] = le_embarked.fit_transform(full['Embarked'])
    full['Title_Enc'] = le_title.fit_transform(full['Title'].astype(str))
    full['Deck_Enc'] = le_deck.fit_transform(full['Deck'].astype(str))
    full['AgeGroup_Enc'] = le_agegroup.fit_transform(full['AgeGroup'].astype(str))
    full['FareGroup_Enc'] = le_faregroup.fit_transform(full['FareGroup'].astype(str))

    # Feature columns (matching V4)
    feature_cols = [
        'Pclass', 'Sex_Enc', 'Age', 'SibSp', 'Parch', 'Fare',
        'Embarked_Enc', 'FamilySize', 'IsAlone', 'Title_Enc',
        'Deck_Enc', 'AgeGroup_Enc', 'FareGroup_Enc',
        'FamilySurvived', 'TicketSurvived', 'GroupSurvived'
    ]

    # Split back
    train_final = full.iloc[:len(train)].copy()
    test_final = full.iloc[len(train):].copy()

    X_train = train_final[feature_cols].values
    y_train = train['Survived'].values
    X_test = test_final[feature_cols].values

    print(f"    Features: {len(feature_cols)}")
    print(f"    X_train shape: {X_train.shape}")

    # =========================================================================
    # TRAIN MODELS (Exact V4 hyperparameters)
    # =========================================================================
    print("\n[4] Training models (V4 hyperparameters)...")

    # XGBoost - EXACTLY as V4
    model_xgb = XGBClassifier(
        n_estimators=100,
        max_depth=3,          # CONSERVATIVE
        learning_rate=0.1,
        gamma=0,
        colsample_bytree=0.8,
        min_child_weight=1,
        subsample=0.8,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Random Forest - matching ranger defaults
    model_rf = RandomForestClassifier(
        n_estimators=500,     # ranger default
        max_features=3,       # mtry=3
        min_samples_leaf=5,   # min.node.size=5
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Logistic Regression with scaling (like glmnet)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_lr = LogisticRegression(
        C=1.0,
        penalty='l2',
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    print("\n    Cross-validation results:")
    cv_xgb = cross_val_score(model_xgb, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"    XGBoost CV: {cv_xgb.mean():.4f} (+/- {cv_xgb.std():.4f})")

    cv_rf = cross_val_score(model_rf, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"    RandomForest CV: {cv_rf.mean():.4f} (+/- {cv_rf.std():.4f})")

    cv_lr = cross_val_score(model_lr, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    print(f"    LogisticReg CV: {cv_lr.mean():.4f} (+/- {cv_lr.std():.4f})")

    # Train on full data
    print("\n[5] Training on full data...")
    model_xgb.fit(X_train, y_train)
    model_rf.fit(X_train, y_train)
    model_lr.fit(X_train_scaled, y_train)

    # =========================================================================
    # ENSEMBLE PREDICTIONS
    # =========================================================================
    print("\n[6] Generating ensemble predictions...")

    # Get probabilities
    prob_xgb = model_xgb.predict_proba(X_test)[:, 1]
    prob_rf = model_rf.predict_proba(X_test)[:, 1]
    prob_lr = model_lr.predict_proba(X_test_scaled)[:, 1]

    # Simple average (exactly as V4)
    final_prob = (prob_xgb + prob_rf + prob_lr) / 3
    final_pred = (final_prob > 0.5).astype(int)

    # =========================================================================
    # SUBMISSION
    # =========================================================================
    survivors = final_pred.sum()
    print(f"\n[7] Results:")
    print(f"    Predicted survivors: {survivors}/{len(test_ids)} ({survivors*100/len(test_ids):.1f}%)")

    # Compare with V4
    v4 = pd.read_csv('2025-R-Attempts/submission_v4.csv')
    match = (final_pred == v4['Survived'].values).sum()
    print(f"    Match with V4: {match}/{len(test_ids)} ({match*100/len(test_ids):.1f}%)")

    # Save submission
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': final_pred
    })
    submission.to_csv('submission_strategy_1.csv', index=False)
    print(f"\n    Saved: submission_strategy_1.csv")

    # Ensemble CV estimate
    ensemble_cv = (cv_xgb.mean() + cv_rf.mean() + cv_lr.mean()) / 3
    print(f"\n    Ensemble CV estimate: {ensemble_cv:.4f}")

    print("\n" + "=" * 60)
    print("STRATEGY 1 COMPLETE")
    print("=" * 60)

    return submission

if __name__ == "__main__":
    main()
