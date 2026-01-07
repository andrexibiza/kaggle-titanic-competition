"""
Approach A: V4 Champion Solution - CORRECTED CV
================================================================
Fixed data leakage in cross-validation for FamilySurvived/TicketSurvived
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def extract_title(name):
    """Extract title from name"""
    title = name.split(',')[1].split('.')[0].strip()
    if title in ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
                 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Mme', 'Mlle', 'Ms']:
        return 'Rare'
    elif title == 'Ms':
        return 'Miss'
    return title

def extract_deck(cabin):
    """Extract deck from cabin"""
    if pd.isna(cabin):
        return 'U'
    return cabin[0]

def engineer_base_features(df):
    """Engineer features that don't depend on target (can be done globally)"""
    df = df.copy()

    # Basic features
    df['Pclass'] = df['Pclass'].astype(int)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Title
    df['Title'] = df['Name'].apply(extract_title)

    # Age imputation by Title
    title_age_median = df.groupby('Title')['Age'].median()
    for title in df['Title'].unique():
        df.loc[(df['Age'].isna()) & (df['Title'] == title), 'Age'] = title_age_median[title]

    # Fare
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Embarked
    df['Embarked'] = df['Embarked'].fillna('S')

    # FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Deck
    df['Deck'] = df['Cabin'].apply(extract_deck)

    # Surname and Ticket for later use
    df['Surname'] = df['Name'].apply(lambda x: x.split(',')[0].strip())

    return df

def add_target_features(df_train, df_apply, default=0.5):
    """
    Add FamilySurvived and TicketSurvived based on df_train survival rates,
    applied to df_apply
    """
    # Calculate from training data only
    family_survival = df_train.groupby('Surname')['Survived'].mean()
    ticket_survival = df_train.groupby('Ticket')['Survived'].mean()

    # Apply to df_apply
    df_apply['FamilySurvived'] = df_apply['Surname'].map(family_survival).fillna(default)
    df_apply['TicketSurvived'] = df_apply['Ticket'].map(ticket_survival).fillna(default)

    return df_apply

def prepare_features(df, label_encoders=None, fit_all_data=None):
    """Encode categorical features"""
    df = df.copy()

    if label_encoders is None:
        # Fit encoders on all data to avoid unseen categories
        if fit_all_data is not None:
            fit_df = fit_all_data
        else:
            fit_df = df

        le_title = LabelEncoder()
        le_title.fit(fit_df['Title'])

        le_deck = LabelEncoder()
        le_deck.fit(fit_df['Deck'])

        le_embarked = LabelEncoder()
        le_embarked.fit(fit_df['Embarked'])

        label_encoders = {
            'Title': le_title,
            'Deck': le_deck,
            'Embarked': le_embarked
        }

    # Transform
    df['Title'] = label_encoders['Title'].transform(df['Title'])
    df['Deck'] = label_encoders['Deck'].transform(df['Deck'])
    df['Embarked'] = label_encoders['Embarked'].transform(df['Embarked'])

    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                    'FamilySize', 'Title', 'Deck', 'FamilySurvived', 'TicketSurvived']

    return df[feature_cols].values, label_encoders, feature_cols

def get_models():
    """Initialize models"""
    xgb = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'
    )
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_split=5,
        min_samples_leaf=2, random_state=42
    )
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    return {'XGBoost': xgb, 'RandomForest': rf, 'LogisticRegression': lr}

def ensemble_predict(models, X):
    """Simple averaging ensemble"""
    probs = [model.predict_proba(X)[:, 1] for model in models.values()]
    avg_prob = np.mean(probs, axis=0)
    return (avg_prob >= 0.5).astype(int), avg_prob

def cross_validate_proper(train_df, n_folds=10):
    """
    Proper cross-validation: FamilySurvived/TicketSurvived calculated
    only from fold training data, not validation data
    """
    # Engineer base features once (independent of target)
    train_processed = engineer_base_features(train_df)

    # Fit label encoders on all data (just for encoding consistency)
    # This doesn't leak information as these are just categorical mappings
    from sklearn.preprocessing import LabelEncoder
    le_title = LabelEncoder()
    le_title.fit(train_processed['Title'])
    le_deck = LabelEncoder()
    le_deck.fit(train_processed['Deck'])
    le_embarked = LabelEncoder()
    le_embarked.fit(train_processed['Embarked'])
    global_encoders = {'Title': le_title, 'Deck': le_deck, 'Embarked': le_embarked}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []

    print(f"\nRunning {n_folds}-Fold Cross-Validation (Proper - No Leakage)...")
    print("=" * 60)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_processed, train_df['Survived']), 1):
        # Split data
        fold_train = train_processed.iloc[train_idx].copy()
        fold_val = train_processed.iloc[val_idx].copy()

        # Add target-dependent features (no leakage - only from fold training)
        fold_train = add_target_features(fold_train, fold_train)
        fold_val = add_target_features(fold_train, fold_val)

        # Prepare features using global encoders
        X_fold_train, _, _ = prepare_features(fold_train, global_encoders)
        X_fold_val, _, _ = prepare_features(fold_val, global_encoders)

        y_fold_train = fold_train['Survived'].values
        y_fold_val = fold_val['Survived'].values

        # Train models
        models = get_models()
        for model in models.values():
            model.fit(X_fold_train, y_fold_train)

        # Predict
        predictions, _ = ensemble_predict(models, X_fold_val)
        accuracy = np.mean(predictions == y_fold_val)
        fold_scores.append(accuracy)

        print(f"Fold {fold:2d}: {accuracy:.5f}")

    print("=" * 60)
    print(f"Mean CV Score: {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")
    print("=" * 60)

    return np.mean(fold_scores), np.std(fold_scores)

def main():
    print("=" * 80)
    print("APPROACH A: V4 Solution - CORRECTED CV (No Data Leakage)")
    print("=" * 80)

    # Load data
    train_df = pd.read_csv('/home/user/kaggle-titanic-competition/train.csv')
    test_df = pd.read_csv('/home/user/kaggle-titanic-competition/test.csv')
    test_ids = test_df['PassengerId'].values

    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Proper cross-validation
    cv_score, cv_std = cross_validate_proper(train_df, n_folds=10)

    # Train final model on full training data
    print("\n" + "=" * 80)
    print("Training final model on full training data...")
    print("=" * 80)

    # Process all data
    train_processed = engineer_base_features(train_df)
    test_processed = engineer_base_features(test_df)

    # Add target features (test uses full training data)
    train_processed = add_target_features(train_processed, train_processed)
    test_processed = add_target_features(train_processed, test_processed)

    # Prepare features
    X_train, encoders, feature_cols = prepare_features(train_processed)
    X_test, _, _ = prepare_features(test_processed, encoders)
    y_train = train_df['Survived'].values

    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # Train models
    models = get_models()
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"  ✓ {name} trained")

    # Predict
    predictions, _ = ensemble_predict(models, X_test)
    survival_rate = np.mean(predictions) * 100

    print(f"\nPredicted Survival Rate: {survival_rate:.2f}% ({np.sum(predictions)}/418)")

    # Save
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': predictions
    })
    submission.to_csv('/home/user/kaggle-titanic-competition/submission_approach_a_corrected.csv', index=False)
    print(f"Submission saved: submission_approach_a_corrected.csv")

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"CV Score (Corrected): {cv_score:.5f} (+/- {cv_std:.5f})")
    print(f"Features:             {len(feature_cols)}")
    print(f"Survival Rate:        {survival_rate:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
