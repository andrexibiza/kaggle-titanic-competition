"""
Approach A: V4 Champion Solution Reproduction (0.78947 target)
================================================================
Architecture: 3-model ensemble (XGBoost + RF + LR) with simple averaging
Features: ~12 carefully engineered features
Conservative hyperparameters, no threshold optimization
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

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def extract_title(name):
    """Extract title from name (Mr, Mrs, Miss, Master, Rare)"""
    title = name.split(',')[1].split('.')[0].strip()
    # Map rare titles
    if title in ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
                 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Mme', 'Mlle', 'Ms']:
        return 'Rare'
    elif title == 'Ms':
        return 'Miss'
    return title

def extract_deck(cabin):
    """Extract deck (first letter of Cabin, U for unknown)"""
    if pd.isna(cabin):
        return 'U'
    return cabin[0]

def calculate_group_survival(df, group_col, default=0.5):
    """Calculate survival rate for groups (family/ticket)"""
    group_survival = df.groupby(group_col)['Survived'].mean()
    return df[group_col].map(group_survival).fillna(default)

def engineer_features(train_df, test_df, add_fare_per_person=False):
    """
    Engineer all V4 features

    Features:
    1. Pclass
    2. Sex
    3. Age (imputed by Title median)
    4. SibSp
    5. Parch
    6. Fare (imputed with median)
    7. Embarked
    8. FamilySize
    9. Title
    10. Deck
    11. FamilySurvived
    12. TicketSurvived
    (Optional) 13. FarePerPerson
    """
    # Combine for consistent feature engineering
    train_len = len(train_df)
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True, sort=False)

    # Create a copy for features
    df = combined.copy()

    # Feature 1-7: Basic features
    df['Pclass'] = df['Pclass'].astype(int)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Feature 9: Title
    df['Title'] = df['Name'].apply(extract_title)

    # Feature 3: Age imputation by Title median
    title_age_median = df.groupby('Title')['Age'].median()
    for title in df['Title'].unique():
        df.loc[(df['Age'].isna()) & (df['Title'] == title), 'Age'] = title_age_median[title]

    # Feature 6: Fare imputation
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Feature 7: Embarked
    df['Embarked'] = df['Embarked'].fillna('S')

    # Feature 8: FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Feature 10: Deck
    df['Deck'] = df['Cabin'].apply(extract_deck)

    # Feature 11: FamilySurvived (surname-based)
    df['Surname'] = df['Name'].apply(lambda x: x.split(',')[0].strip())

    # Calculate family survival only from training data
    train_part = df.iloc[:train_len].copy()
    family_survival = train_part.groupby('Surname')['Survived'].mean()
    df['FamilySurvived'] = df['Surname'].map(family_survival).fillna(0.5)

    # Feature 12: TicketSurvived
    ticket_survival = train_part.groupby('Ticket')['Survived'].mean()
    df['TicketSurvived'] = df['Ticket'].map(ticket_survival).fillna(0.5)

    # Optional Feature 13: FarePerPerson
    if add_fare_per_person:
        ticket_counts = df.groupby('Ticket')['PassengerId'].count()
        df['TicketGroupSize'] = df['Ticket'].map(ticket_counts)
        df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']

    # Encode categorical features
    le_title = LabelEncoder()
    df['Title'] = le_title.fit_transform(df['Title'])

    le_deck = LabelEncoder()
    df['Deck'] = le_deck.fit_transform(df['Deck'])

    le_embarked = LabelEncoder()
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

    # Select features
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                    'FamilySize', 'Title', 'Deck', 'FamilySurvived', 'TicketSurvived']

    if add_fare_per_person:
        feature_cols.append('FarePerPerson')

    # Split back
    X_train = df.iloc[:train_len][feature_cols].values
    X_test = df.iloc[train_len:][feature_cols].values
    y_train = train_df['Survived'].values

    return X_train, X_test, y_train, feature_cols

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

def get_models():
    """Initialize the 3 models with V4 conservative hyperparameters"""
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    )

    return {'XGBoost': xgb, 'RandomForest': rf, 'LogisticRegression': lr}

# ============================================================================
# ENSEMBLE & EVALUATION
# ============================================================================

def ensemble_predict(models, X):
    """Simple averaging ensemble"""
    probs = []
    for model in models.values():
        prob = model.predict_proba(X)[:, 1]
        probs.append(prob)

    # Average probabilities
    avg_prob = np.mean(probs, axis=0)

    # Threshold at 0.5
    predictions = (avg_prob >= 0.5).astype(int)

    return predictions, avg_prob

def cross_validate(X, y, n_folds=10):
    """10-fold stratified cross-validation"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_scores = []

    print(f"\nRunning {n_folds}-Fold Cross-Validation...")
    print("=" * 60)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        # Train all models
        models = get_models()
        for name, model in models.items():
            model.fit(X_fold_train, y_fold_train)

        # Ensemble prediction
        predictions, _ = ensemble_predict(models, X_fold_val)

        # Calculate accuracy
        accuracy = np.mean(predictions == y_fold_val)
        fold_scores.append(accuracy)

        print(f"Fold {fold:2d}: {accuracy:.5f}")

    print("=" * 60)
    print(f"Mean CV Score: {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")
    print("=" * 60)

    return np.mean(fold_scores), np.std(fold_scores)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("APPROACH A: V4 Champion Solution Reproduction")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('/home/user/kaggle-titanic-competition/train.csv')
    test_df = pd.read_csv('/home/user/kaggle-titanic-competition/test.csv')
    test_ids = test_df['PassengerId'].values

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # ========================================================================
    # PHASE 1: V4 Baseline (12 features)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: V4 Baseline (12 features)")
    print("=" * 80)

    X_train, X_test, y_train, feature_cols = engineer_features(train_df, test_df)

    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # Cross-validation
    cv_score, cv_std = cross_validate(X_train, y_train, n_folds=10)

    # Train final models on full training data
    print("\nTraining final models on full training data...")
    models = get_models()
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"  ✓ {name} trained")

    # Generate predictions
    predictions, probabilities = ensemble_predict(models, X_test)

    # Calculate survival rate
    survival_rate = np.mean(predictions) * 100
    survival_count = np.sum(predictions)

    print(f"\nPredicted Survival Rate: {survival_rate:.2f}% ({survival_count}/418)")

    # Save submission
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': predictions
    })
    submission.to_csv('/home/user/kaggle-titanic-competition/submission_approach_a.csv', index=False)
    print(f"Submission saved: submission_approach_a.csv")

    # ========================================================================
    # PHASE 2: Test FarePerPerson Feature (13 features)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: Testing FarePerPerson Feature (13 features)")
    print("=" * 80)

    X_train_fpp, X_test_fpp, y_train_fpp, feature_cols_fpp = engineer_features(
        train_df, test_df, add_fare_per_person=True
    )

    print(f"\nFeatures ({len(feature_cols_fpp)}): {feature_cols_fpp}")

    # Cross-validation with FarePerPerson
    cv_score_fpp, cv_std_fpp = cross_validate(X_train_fpp, y_train_fpp, n_folds=10)

    # Compare results
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print(f"V4 Baseline (12 features):  CV = {cv_score:.5f} (+/- {cv_std:.5f})")
    print(f"With FarePerPerson (13 f):  CV = {cv_score_fpp:.5f} (+/- {cv_std_fpp:.5f})")

    improvement = cv_score_fpp - cv_score
    print(f"\nImprovement: {improvement:+.5f}")

    if cv_score_fpp > cv_score:
        print("✓ FarePerPerson IMPROVES performance - keeping it")

        # Generate new submission with FarePerPerson
        print("\nTraining final models with FarePerPerson...")
        models_fpp = get_models()
        for name, model in models_fpp.items():
            model.fit(X_train_fpp, y_train_fpp)

        predictions_fpp, _ = ensemble_predict(models_fpp, X_test_fpp)
        survival_rate_fpp = np.mean(predictions_fpp) * 100
        survival_count_fpp = np.sum(predictions_fpp)

        print(f"Predicted Survival Rate: {survival_rate_fpp:.2f}% ({survival_count_fpp}/418)")

        submission_fpp = pd.DataFrame({
            'PassengerId': test_ids,
            'Survived': predictions_fpp
        })
        submission_fpp.to_csv('/home/user/kaggle-titanic-competition/submission_approach_a_refined.csv', index=False)
        print(f"Refined submission saved: submission_approach_a_refined.csv")

        final_cv = cv_score_fpp
        final_features = len(feature_cols_fpp)
        final_survival = survival_rate_fpp
    else:
        print("✗ FarePerPerson does NOT improve performance - discarding")
        final_cv = cv_score
        final_features = len(feature_cols)
        final_survival = survival_rate

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - APPROACH A")
    print("=" * 80)
    print(f"Architecture:     3-model ensemble (XGBoost + RF + LR)")
    print(f"Ensemble Method:  Simple averaging")
    print(f"Threshold:        0.5 (standard)")
    print(f"Features:         {final_features}")
    print(f"CV Score:         {final_cv:.5f}")
    print(f"Survival Rate:    {final_survival:.2f}%")
    print(f"Target:           ~37% survival, 0.789+ accuracy")
    print("=" * 80)

    return {
        'cv_score': final_cv,
        'n_features': final_features,
        'survival_rate': final_survival,
        'fare_per_person_helps': cv_score_fpp > cv_score
    }

if __name__ == "__main__":
    results = main()
