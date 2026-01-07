#!/usr/bin/env python3
"""
Approach C: Seed-Averaged Ensemble for Kaggle Titanic Competition
===================================================================

GOAL: Build a seed-averaged ensemble to reduce variance while capturing signal.

ARCHITECTURE:
- Train V4-style ensemble (XGB + RF + LR) across 10 different seeds
- Average all predictions
- Use consensus voting for high-confidence cases

SEEDS: [42, 123, 456, 789, 1000, 2023, 2024, 2025, 314, 271]

FEATURES (same as V4, ~12):
1. Pclass
2. Sex (encoded)
3. Age (imputed by Title median)
4. SibSp, Parch
5. Fare
6. Embarked (encoded)
7. FamilySize
8. Title (encoded)
9. FamilySurvived (default 0.5)
10. TicketSurvived (default 0.5)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import re
from typing import Tuple, List

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Seeds for ensemble
SEEDS = [42, 123, 456, 789, 1000, 2023, 2024, 2025, 314, 271]
N_FOLDS = 10


class TitanicFeatureEngineer:
    """Simple feature engineering for V4-style features."""

    def __init__(self):
        self.age_medians = {}
        self.family_survival = {}
        self.ticket_survival = {}

    def extract_title(self, name: str) -> str:
        """Extract title from passenger name."""
        title_search = re.search(r' ([A-Za-z]+)\.', name)
        if title_search:
            return title_search.group(1)
        return "Unknown"

    def map_title(self, title: str) -> str:
        """Map rare titles to common categories."""
        title_dict = {
            'Mr': 'Mr',
            'Miss': 'Miss',
            'Mrs': 'Mrs',
            'Master': 'Master',
            'Mlle': 'Miss',
            'Ms': 'Miss',
            'Mme': 'Mrs',
        }
        return title_dict.get(title, 'Rare')

    def extract_surname(self, name: str) -> str:
        """Extract surname from passenger name."""
        return name.split(',')[0].strip()

    def fit(self, train_df: pd.DataFrame) -> 'TitanicFeatureEngineer':
        """Fit the feature engineer on training data."""
        df = train_df.copy()

        # Extract titles
        df['Title'] = df['Name'].apply(self.extract_title)
        df['Title_Mapped'] = df['Title'].apply(self.map_title)

        # Calculate age medians by Title
        for title in df['Title_Mapped'].unique():
            median_age = df.loc[df['Title_Mapped'] == title, 'Age'].median()
            if pd.isna(median_age):
                median_age = df['Age'].median()
            self.age_medians[title] = median_age

        # Calculate family survival rates (surname + fare as proxy)
        df['Surname'] = df['Name'].apply(self.extract_surname)
        df['FareRange'] = pd.cut(df['Fare'], bins=[0, 10, 20, 30, 100, 600], labels=[0, 1, 2, 3, 4])
        df['FamilyID'] = df['Surname'] + '_' + df['FareRange'].astype(str)

        family_stats = df.groupby('FamilyID').agg({
            'Survived': ['mean', 'count']
        }).reset_index()
        family_stats.columns = ['FamilyID', 'SurvivalRate', 'Count']

        for _, row in family_stats.iterrows():
            if row['Count'] >= 2:
                self.family_survival[row['FamilyID']] = row['SurvivalRate']

        # Calculate ticket survival rates
        ticket_stats = df.groupby('Ticket').agg({
            'Survived': ['mean', 'count']
        }).reset_index()
        ticket_stats.columns = ['Ticket', 'SurvivalRate', 'Count']

        for _, row in ticket_stats.iterrows():
            if row['Count'] >= 2:
                self.ticket_survival[row['Ticket']] = row['SurvivalRate']

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe with engineered features."""
        result = df.copy()

        # Extract title
        result['Title'] = result['Name'].apply(self.extract_title)
        result['Title_Mapped'] = result['Title'].apply(self.map_title)

        # Encode sex
        result['Sex_Code'] = result['Sex'].map({'female': 1, 'male': 0}).astype(int)

        # Impute age by title median
        for title in result['Title_Mapped'].unique():
            mask = (result['Title_Mapped'] == title) & (result['Age'].isna())
            median_age = self.age_medians.get(title, result['Age'].median())
            result.loc[mask, 'Age'] = median_age

        # Fill remaining age NaNs
        result['Age'] = result['Age'].fillna(result['Age'].median())

        # Fill fare NaNs
        result['Fare'] = result['Fare'].fillna(result['Fare'].median())

        # Encode embarked
        result['Embarked'] = result['Embarked'].fillna('S')
        result['Embarked_Code'] = result['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        # Family size
        result['FamilySize'] = result['SibSp'] + result['Parch'] + 1

        # Title encoding
        title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
        result['Title_Code'] = result['Title_Mapped'].map(title_mapping).fillna(0)

        # Family survival
        result['Surname'] = result['Name'].apply(self.extract_surname)
        result['FareRange'] = pd.cut(result['Fare'], bins=[0, 10, 20, 30, 100, 600], labels=[0, 1, 2, 3, 4])
        result['FamilyID'] = result['Surname'] + '_' + result['FareRange'].astype(str)
        result['FamilySurvived'] = result['FamilyID'].map(self.family_survival).fillna(0.5)

        # Ticket survival
        result['TicketSurvived'] = result['Ticket'].map(self.ticket_survival).fillna(0.5)

        return result


def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load and prepare train/test data."""
    print("Loading data...")
    train = pd.read_csv('/home/user/kaggle-titanic-competition/train.csv')
    test = pd.read_csv('/home/user/kaggle-titanic-competition/test.csv')

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    # Feature engineering
    print("Engineering features...")
    engineer = TitanicFeatureEngineer()
    engineer.fit(train)

    train_processed = engineer.transform(train)
    test_processed = engineer.transform(test)

    # Select features (12 features)
    features = [
        'Pclass', 'Sex_Code', 'Age', 'SibSp', 'Parch', 'Fare',
        'Embarked_Code', 'FamilySize', 'Title_Code',
        'FamilySurvived', 'TicketSurvived'
    ]

    X_train = train_processed[features].values
    y_train = train['Survived'].values
    X_test = test_processed[features].values

    print(f"Features ({len(features)}): {features}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    return X_train, y_train, X_test, test


def train_seed_ensemble(X_train: np.ndarray, y_train: np.ndarray,
                        seed: int) -> Tuple[XGBClassifier, RandomForestClassifier, LogisticRegression]:
    """Train XGB, RF, and LR for a given seed."""

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric='logloss'
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=seed
    )

    # Logistic Regression
    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=seed
    )

    # Train models
    xgb.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    return xgb, rf, lr


def predict_seed_ensemble(models: Tuple, X: np.ndarray) -> np.ndarray:
    """Average predictions from XGB, RF, LR."""
    xgb, rf, lr = models

    prob_xgb = xgb.predict_proba(X)[:, 1]
    prob_rf = rf.predict_proba(X)[:, 1]
    prob_lr = lr.predict_proba(X)[:, 1]

    # Average probabilities
    avg_prob = (prob_xgb + prob_rf + prob_lr) / 3

    return avg_prob


def run_cv_with_seed_ensemble(X_train: np.ndarray, y_train: np.ndarray,
                               n_folds: int = 10) -> Tuple[float, dict]:
    """Run cross-validation with seed-averaged ensemble."""

    print(f"\n{'='*60}")
    print(f"Running {n_folds}-Fold Cross-Validation with Seed Ensemble")
    print(f"{'='*60}\n")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_scores = []
    seed_variances = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\nFold {fold_idx + 1}/{n_folds}")
        print("-" * 40)

        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]

        # Collect predictions from all seeds
        all_seed_probs = []

        for seed in SEEDS:
            # Train models for this seed
            models = train_seed_ensemble(X_fold_train, y_fold_train, seed)

            # Get averaged probabilities for this seed
            seed_prob = predict_seed_ensemble(models, X_fold_val)
            all_seed_probs.append(seed_prob)

        # Convert to array: shape (n_seeds, n_samples)
        all_seed_probs = np.array(all_seed_probs)

        # Average across all seeds
        final_prob = np.mean(all_seed_probs, axis=0)
        final_pred = (final_prob >= 0.5).astype(int)

        # Calculate variance across seeds
        seed_variance = np.var(all_seed_probs, axis=0).mean()
        seed_variances.append(seed_variance)

        # Calculate accuracy
        accuracy = accuracy_score(y_fold_val, final_pred)
        fold_scores.append(accuracy)

        print(f"Fold {fold_idx + 1} Accuracy: {accuracy:.4f}")
        print(f"Fold {fold_idx + 1} Seed Variance: {seed_variance:.6f}")

    mean_cv_score = np.mean(fold_scores)
    std_cv_score = np.std(fold_scores)
    mean_variance = np.mean(seed_variances)

    print(f"\n{'='*60}")
    print(f"Cross-Validation Results")
    print(f"{'='*60}")
    print(f"Mean CV Accuracy: {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")
    print(f"Mean Seed Variance: {mean_variance:.6f}")
    print(f"Individual Fold Scores: {[f'{s:.4f}' for s in fold_scores]}")

    variance_stats = {
        'mean_variance': mean_variance,
        'std_variance': np.std(seed_variances),
        'fold_variances': seed_variances
    }

    return mean_cv_score, variance_stats


def generate_submission(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, test_df: pd.DataFrame) -> pd.DataFrame:
    """Generate final submission using seed-averaged ensemble."""

    print(f"\n{'='*60}")
    print("Generating Final Submission")
    print(f"{'='*60}\n")

    all_seed_probs = []

    for seed_idx, seed in enumerate(SEEDS):
        print(f"Training with seed {seed} ({seed_idx + 1}/{len(SEEDS)})...")

        # Train models for this seed
        models = train_seed_ensemble(X_train, y_train, seed)

        # Get averaged probabilities for this seed
        seed_prob = predict_seed_ensemble(models, X_test)
        all_seed_probs.append(seed_prob)

    # Convert to array: shape (n_seeds, n_samples)
    all_seed_probs = np.array(all_seed_probs)

    # Average across all seeds
    final_prob = np.mean(all_seed_probs, axis=0)
    final_pred = (final_prob >= 0.5).astype(int)

    # Calculate variance across seeds for test set
    test_variance = np.var(all_seed_probs, axis=0)

    print(f"\nTest Set Statistics:")
    print(f"Mean probability: {final_prob.mean():.4f}")
    print(f"Mean seed variance: {test_variance.mean():.6f}")
    print(f"Max seed variance: {test_variance.max():.6f}")
    print(f"Min seed variance: {test_variance.min():.6f}")

    # Survival statistics
    n_survived = final_pred.sum()
    n_total = len(final_pred)
    survival_rate = n_survived / n_total

    print(f"\nPredicted Survival Statistics:")
    print(f"Survived: {n_survived}/{n_total} ({survival_rate:.2%})")

    # Verify survival rate
    if survival_rate < 0.36 or survival_rate > 0.38:
        print(f"\nWARNING: Survival rate {survival_rate:.2%} is outside target range (36-38%)")
    else:
        print(f"\nSurvival rate {survival_rate:.2%} is within target range (36-38%)")

    # Create submission
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': final_pred
    })

    return submission, final_prob, test_variance


def main():
    """Main execution function."""

    print(f"\n{'='*60}")
    print("Approach C: Seed-Averaged Ensemble")
    print(f"{'='*60}\n")

    print(f"Configuration:")
    print(f"  - Number of seeds: {len(SEEDS)}")
    print(f"  - Seeds: {SEEDS}")
    print(f"  - Models per seed: XGBoost + Random Forest + Logistic Regression")
    print(f"  - CV folds: {N_FOLDS}")

    # Load and prepare data
    X_train, y_train, X_test, test_df = load_and_prepare_data()

    # Run cross-validation
    cv_score, variance_stats = run_cv_with_seed_ensemble(X_train, y_train, N_FOLDS)

    # Generate submission
    submission, final_prob, test_variance = generate_submission(
        X_train, y_train, X_test, test_df
    )

    # Save submission
    submission_path = '/home/user/kaggle-titanic-competition/submission_approach_c.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"CV Accuracy: {cv_score:.4f}")
    print(f"Mean Seed Variance (CV): {variance_stats['mean_variance']:.6f}")
    print(f"Mean Seed Variance (Test): {test_variance.mean():.6f}")
    print(f"Predicted Survival Count: {submission['Survived'].sum()}")
    print(f"Predicted Survival Rate: {submission['Survived'].mean():.2%}")
    print(f"{'='*60}\n")

    return cv_score, variance_stats, submission


if __name__ == "__main__":
    cv_score, variance_stats, submission = main()
