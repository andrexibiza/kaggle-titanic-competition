#!/usr/bin/env python3
"""
Advanced Titanic Survival Prediction Solution
=============================================

Target: 80-85% accuracy on Kaggle leaderboard

Key innovations:
1. Advanced Feature Engineering with interaction features
2. Probabilistic WCG (Woman-Child-Group) with confidence blending
3. Multi-model ensemble: CatBoost, XGBoost, LightGBM, RF, ExtraTrees
4. Two-level stacking with optimized meta-learner
5. Optuna Bayesian hyperparameter optimization
6. Threshold optimization for decision boundary

Author: Advanced ML Pipeline
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import re
from collections import defaultdict

# Scikit-learn
from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# Gradient Boosting Libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Hyperparameter Optimization
import optuna
from optuna.samplers import TPESampler

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class TitanicFeatureEngineer:
    """Advanced feature engineering for Titanic dataset."""

    def __init__(self):
        self.title_mapping = {}
        self.surname_survival = {}
        self.ticket_survival = {}
        self.age_medians = {}
        self.fare_medians = {}
        self.deck_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8, 'U': 0}
        self.embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}

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
            'Dr': 'Officer',
            'Rev': 'Officer',
            'Col': 'Officer',
            'Major': 'Officer',
            'Capt': 'Officer',
            'Mlle': 'Miss',
            'Ms': 'Miss',
            'Mme': 'Mrs',
            'Don': 'Royalty',
            'Dona': 'Royalty',
            'Lady': 'Royalty',
            'Sir': 'Royalty',
            'Countess': 'Royalty',
            'Jonkheer': 'Royalty',
        }
        return title_dict.get(title, 'Rare')

    def extract_surname(self, name: str) -> str:
        """Extract surname from passenger name."""
        return name.split(',')[0].strip()

    def extract_deck(self, cabin: str) -> str:
        """Extract deck from cabin number."""
        if pd.isna(cabin) or cabin == '':
            return 'U'
        return cabin[0]

    def fit(self, train_df: pd.DataFrame) -> 'TitanicFeatureEngineer':
        """Fit the feature engineer on training data."""
        df = train_df.copy()

        # Extract basic features first
        df['Title'] = df['Name'].apply(self.extract_title)
        df['Title_Mapped'] = df['Title'].apply(self.map_title)
        df['Surname'] = df['Name'].apply(self.extract_surname)

        # Calculate age medians by Title and Pclass
        for title in df['Title_Mapped'].unique():
            for pclass in df['Pclass'].unique():
                mask = (df['Title_Mapped'] == title) & (df['Pclass'] == pclass)
                median_age = df.loc[mask & df['Age'].notna(), 'Age'].median()
                if pd.isna(median_age):
                    median_age = df.loc[df['Age'].notna(), 'Age'].median()
                self.age_medians[(title, pclass)] = median_age

        # Calculate fare medians by Pclass
        for pclass in df['Pclass'].unique():
            self.fare_medians[pclass] = df.loc[df['Pclass'] == pclass, 'Fare'].median()

        # Calculate surname survival rates from training data
        surname_stats = df.groupby('Surname').agg({
            'Survived': ['mean', 'count']
        }).reset_index()
        surname_stats.columns = ['Surname', 'SurvivalRate', 'Count']

        for _, row in surname_stats.iterrows():
            if row['Count'] >= 2:  # Only for families with 2+ members
                self.surname_survival[row['Surname']] = {
                    'rate': row['SurvivalRate'],
                    'count': row['Count'],
                    'confidence': min(1.0, row['Count'] / 5)  # Confidence scaling
                }

        # Calculate ticket survival rates from training data
        ticket_stats = df.groupby('Ticket').agg({
            'Survived': ['mean', 'count']
        }).reset_index()
        ticket_stats.columns = ['Ticket', 'SurvivalRate', 'Count']

        for _, row in ticket_stats.iterrows():
            if row['Count'] >= 2:  # Only for groups with 2+ members
                self.ticket_survival[row['Ticket']] = {
                    'rate': row['SurvivalRate'],
                    'count': row['Count'],
                    'confidence': min(1.0, row['Count'] / 5)
                }

        return self

    def transform(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """Transform dataframe with advanced features."""
        df = df.copy()

        # 1. BASIC FEATURE EXTRACTION
        df['Title'] = df['Name'].apply(self.extract_title)
        df['Title_Mapped'] = df['Title'].apply(self.map_title)
        df['Surname'] = df['Name'].apply(self.extract_surname)
        df['Deck'] = df['Cabin'].apply(self.extract_deck)

        # 2. TITLE ENCODING
        title_encoder = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Officer': 4, 'Royalty': 5, 'Rare': 6}
        df['Title_Encoded'] = df['Title_Mapped'].map(title_encoder).fillna(6)

        # 3. IS_BOY FLAG (Master title = young boys, very important feature)
        df['IsBoy'] = (df['Title_Mapped'] == 'Master').astype(int)

        # 4. IS_WOMAN FLAG
        df['IsWoman'] = (df['Sex'] == 'female').astype(int)

        # 5. IS_WOMAN_OR_CHILD
        df['IsWomanOrChild'] = ((df['Sex'] == 'female') | (df['Title_Mapped'] == 'Master')).astype(int)

        # 6. SEX ENCODING
        df['Sex_Encoded'] = (df['Sex'] == 'male').astype(int)

        # 7. AGE IMPUTATION (stratified by Title and Pclass)
        for idx in df[df['Age'].isna()].index:
            title = df.loc[idx, 'Title_Mapped']
            pclass = df.loc[idx, 'Pclass']
            key = (title, pclass)
            if key in self.age_medians:
                df.loc[idx, 'Age'] = self.age_medians[key]
            else:
                df.loc[idx, 'Age'] = df['Age'].median()

        # 8. AGE GROUPS (non-linear binning)
        df['AgeGroup'] = pd.cut(df['Age'],
                                bins=[0, 5, 12, 18, 35, 50, 65, 100],
                                labels=[0, 1, 2, 3, 4, 5, 6]).astype(float)
        df['AgeGroup'] = df['AgeGroup'].fillna(3)

        # 9. IS_CHILD FLAG
        df['IsChild'] = (df['Age'] < 12).astype(int)

        # 10. FARE IMPUTATION AND PROCESSING
        for idx in df[df['Fare'].isna()].index:
            pclass = df.loc[idx, 'Pclass']
            df.loc[idx, 'Fare'] = self.fare_medians.get(pclass, df['Fare'].median())

        # 11. TICKET GROUP SIZE
        ticket_counts = df['Ticket'].value_counts()
        df['TicketGroupSize'] = df['Ticket'].map(ticket_counts)

        # 12. FARE PER PERSON (corrected for group bookings)
        df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']
        df['FarePerPerson'] = df['FarePerPerson'].replace([np.inf, -np.inf], 0).fillna(0)

        # 13. FARE BANDS
        df['FareBand'] = pd.qcut(df['FarePerPerson'].clip(lower=0.01),
                                  q=5, labels=[0, 1, 2, 3, 4], duplicates='drop').astype(float)
        df['FareBand'] = df['FareBand'].fillna(2)

        # 14. FAMILY SIZE FEATURES
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['SmallFamily'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)).astype(int)
        df['LargeFamily'] = (df['FamilySize'] > 4).astype(int)

        # 15. FAMILY SIZE BINNED (non-linear)
        df['FamilySizeBinned'] = df['FamilySize'].apply(
            lambda x: 0 if x == 1 else (1 if x <= 4 else 2)
        )

        # 16. SURNAME SURVIVAL FEATURES
        df['SurnameSurvivalRate'] = df['Surname'].apply(
            lambda x: self.surname_survival.get(x, {}).get('rate', 0.5)
        )
        df['SurnameFamilySize'] = df['Surname'].apply(
            lambda x: self.surname_survival.get(x, {}).get('count', 1)
        )
        df['SurnameConfidence'] = df['Surname'].apply(
            lambda x: self.surname_survival.get(x, {}).get('confidence', 0)
        )

        # 17. TICKET SURVIVAL FEATURES
        df['TicketSurvivalRate'] = df['Ticket'].apply(
            lambda x: self.ticket_survival.get(x, {}).get('rate', 0.5)
        )
        df['TicketFamilySize'] = df['Ticket'].apply(
            lambda x: self.ticket_survival.get(x, {}).get('count', 1)
        )
        df['TicketConfidence'] = df['Ticket'].apply(
            lambda x: self.ticket_survival.get(x, {}).get('confidence', 0)
        )

        # 18. GROUP SURVIVAL (max of surname and ticket)
        df['GroupSurvivalRate'] = df[['SurnameSurvivalRate', 'TicketSurvivalRate']].max(axis=1)
        df['GroupConfidence'] = df[['SurnameConfidence', 'TicketConfidence']].max(axis=1)

        # 19. WCG SCORE (key feature for probabilistic WCG)
        # Higher score = more likely to survive based on group patterns
        df['WCG_Score'] = np.where(
            df['IsWomanOrChild'] == 1,
            df['GroupSurvivalRate'] * df['GroupConfidence'],
            0.5 - (1 - df['GroupSurvivalRate']) * df['GroupConfidence']  # Inverse for men
        )

        # 20. DECK ENCODING
        df['Deck_Encoded'] = df['Deck'].map(self.deck_mapping).fillna(0)

        # 21. HAS CABIN FLAG
        df['HasCabin'] = (df['Deck'] != 'U').astype(int)

        # 22. EMBARKED ENCODING
        df['Embarked'] = df['Embarked'].fillna('S')
        df['Embarked_Encoded'] = df['Embarked'].map(self.embarked_mapping)

        # 23. NAME LENGTH (proxy for social status/prominence)
        df['NameLength'] = df['Name'].apply(len)

        # 24. TICKET PREFIX FEATURES
        df['TicketPrefix'] = df['Ticket'].apply(
            lambda x: x.split()[0] if len(x.split()) > 1 else 'NONE'
        )
        df['HasTicketPrefix'] = (df['TicketPrefix'] != 'NONE').astype(int)

        # 25. INTERACTION FEATURES
        df['Pclass_Sex'] = df['Pclass'] * 10 + df['Sex_Encoded']
        df['Pclass_Age'] = df['Pclass'] * df['Age']
        df['Pclass_Fare'] = df['Pclass'] * df['FarePerPerson']
        df['Age_Sex'] = df['Age'] * df['Sex_Encoded']
        df['FamilySize_Pclass'] = df['FamilySize'] * df['Pclass']

        # 26. TITLE-PCLASS INTERACTION (very important)
        df['Title_Pclass'] = df['Title_Encoded'] * 10 + df['Pclass']

        # 27. WOMEN IN 3RD CLASS (specific vulnerable group)
        df['Woman_3rdClass'] = ((df['Sex'] == 'female') & (df['Pclass'] == 3)).astype(int)

        # 28. MEN IN 1ST CLASS (specific survival group among men)
        df['Man_1stClass'] = ((df['Sex'] == 'male') & (df['Pclass'] == 1)).astype(int)

        # 29. CHILD WITH FAMILY (children traveling with family have better survival)
        df['Child_WithFamily'] = ((df['IsChild'] == 1) & (df['IsAlone'] == 0)).astype(int)

        return df

    def get_feature_columns(self) -> List[str]:
        """Return list of feature columns for model training."""
        return [
            'Pclass', 'Sex_Encoded', 'Age', 'SibSp', 'Parch', 'FarePerPerson',
            'Title_Encoded', 'IsBoy', 'IsWoman', 'IsWomanOrChild', 'AgeGroup',
            'IsChild', 'FareBand', 'FamilySize', 'IsAlone', 'SmallFamily',
            'LargeFamily', 'FamilySizeBinned', 'SurnameSurvivalRate',
            'SurnameConfidence', 'TicketSurvivalRate', 'TicketConfidence',
            'GroupSurvivalRate', 'GroupConfidence', 'WCG_Score',
            'Deck_Encoded', 'HasCabin', 'Embarked_Encoded', 'NameLength',
            'HasTicketPrefix', 'TicketGroupSize', 'Pclass_Sex', 'Pclass_Age',
            'Age_Sex', 'FamilySize_Pclass', 'Title_Pclass', 'Woman_3rdClass',
            'Man_1stClass', 'Child_WithFamily'
        ]


class ProbabilisticWCG:
    """
    Probabilistic Woman-Child-Group (WCG) Logic.

    Instead of hard overrides, uses confidence-weighted blending of
    group survival patterns with model predictions.
    """

    def __init__(self, threshold_confidence: float = 0.8):
        self.threshold_confidence = threshold_confidence
        self.ticket_groups = {}
        self.surname_groups = {}

    def fit(self, train_df: pd.DataFrame) -> 'ProbabilisticWCG':
        """Learn group survival patterns from training data."""
        df = train_df.copy()

        # Extract surnames
        df['Surname'] = df['Name'].apply(lambda x: x.split(',')[0].strip())

        # Ticket group analysis
        for ticket, group in df.groupby('Ticket'):
            if len(group) >= 2:
                survival_rate = group['Survived'].mean()
                all_survived = survival_rate == 1.0
                all_died = survival_rate == 0.0

                self.ticket_groups[ticket] = {
                    'survival_rate': survival_rate,
                    'count': len(group),
                    'all_survived': all_survived,
                    'all_died': all_died,
                    'deterministic': all_survived or all_died,
                    'confidence': min(1.0, len(group) / 4)
                }

        # Surname group analysis
        for surname, group in df.groupby('Surname'):
            if len(group) >= 2:
                survival_rate = group['Survived'].mean()
                all_survived = survival_rate == 1.0
                all_died = survival_rate == 0.0

                self.surname_groups[surname] = {
                    'survival_rate': survival_rate,
                    'count': len(group),
                    'all_survived': all_survived,
                    'all_died': all_died,
                    'deterministic': all_survived or all_died,
                    'confidence': min(1.0, len(group) / 4)
                }

        return self

    def get_wcg_predictions(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get WCG-based predictions and confidence scores.

        Returns:
            wcg_predictions: Array of WCG predictions (0, 1, or NaN for uncertain)
            wcg_confidences: Array of confidence scores [0, 1]
        """
        df = test_df.copy()
        df['Surname'] = df['Name'].apply(lambda x: x.split(',')[0].strip())

        # Extract title to identify women and children
        df['Title'] = df['Name'].apply(lambda x: re.search(r' ([A-Za-z]+)\.', x).group(1) if re.search(r' ([A-Za-z]+)\.', x) else 'Unknown')
        df['IsWomanOrChild'] = (df['Sex'] == 'female') | (df['Title'] == 'Master')

        n = len(df)
        wcg_predictions = np.full(n, np.nan)
        wcg_confidences = np.zeros(n)

        for idx, row in df.iterrows():
            pos = df.index.get_loc(idx)

            # Only apply WCG to women and children
            if not row['IsWomanOrChild']:
                wcg_predictions[pos] = np.nan
                wcg_confidences[pos] = 0
                continue

            ticket = row['Ticket']
            surname = row['Surname']

            # Check ticket group first (higher priority)
            if ticket in self.ticket_groups:
                group_info = self.ticket_groups[ticket]
                if group_info['deterministic']:
                    wcg_predictions[pos] = 1 if group_info['all_survived'] else 0
                    wcg_confidences[pos] = group_info['confidence']
                    continue

            # Check surname group second
            if surname in self.surname_groups:
                group_info = self.surname_groups[surname]
                if group_info['deterministic']:
                    wcg_predictions[pos] = 1 if group_info['all_survived'] else 0
                    wcg_confidences[pos] = group_info['confidence']
                    continue

            # For mixed groups, use probabilistic approach
            best_confidence = 0
            if ticket in self.ticket_groups:
                group_info = self.ticket_groups[ticket]
                if group_info['confidence'] > best_confidence:
                    best_confidence = group_info['confidence']
                    wcg_predictions[pos] = group_info['survival_rate']
                    wcg_confidences[pos] = group_info['confidence'] * 0.5  # Reduced confidence for non-deterministic

            if surname in self.surname_groups:
                group_info = self.surname_groups[surname]
                if group_info['confidence'] > best_confidence:
                    wcg_predictions[pos] = group_info['survival_rate']
                    wcg_confidences[pos] = group_info['confidence'] * 0.5

        return wcg_predictions, wcg_confidences

    def blend_predictions(self,
                          ml_predictions: np.ndarray,
                          wcg_predictions: np.ndarray,
                          wcg_confidences: np.ndarray,
                          blend_strength: float = 0.7) -> np.ndarray:
        """
        Blend ML predictions with WCG predictions based on confidence.

        Args:
            ml_predictions: Model probability predictions
            wcg_predictions: WCG probability predictions
            wcg_confidences: Confidence scores for WCG predictions
            blend_strength: How much to weight WCG vs ML (0-1)

        Returns:
            Blended probability predictions
        """
        blended = np.copy(ml_predictions)

        for i in range(len(blended)):
            if not np.isnan(wcg_predictions[i]) and wcg_confidences[i] > 0:
                # Weighted blend based on confidence
                weight = wcg_confidences[i] * blend_strength
                blended[i] = weight * wcg_predictions[i] + (1 - weight) * ml_predictions[i]

        return blended


class OptimizedEnsemble:
    """
    Multi-model ensemble with Bayesian hyperparameter optimization.
    """

    def __init__(self, n_trials: int = 25, cv_folds: int = 5, random_state: int = 42):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_params = {}
        self.models = {}
        self.model_weights = {}
        self.scaler = StandardScaler()

    def _objective_xgb(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optuna objective for XGBoost."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2),
            'random_state': self.random_state,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }

        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model.fit(X_train, y_train, verbose=False)
            pred = model.predict(X_val)
            scores.append(accuracy_score(y_val, pred))

        return np.mean(scores)

    def _objective_lgbm(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optuna objective for LightGBM."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': self.random_state,
            'verbosity': -1
        }

        model = LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            scores.append(accuracy_score(y_val, pred))

        return np.mean(scores)

    def _objective_catboost(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optuna objective for CatBoost."""
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_seed': self.random_state,
            'verbose': False
        }

        model = CatBoostClassifier(**params)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            scores.append(accuracy_score(y_val, pred))

        return np.mean(scores)

    def _objective_rf(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optuna objective for Random Forest."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state': self.random_state,
            'n_jobs': 1
        }

        model = RandomForestClassifier(**params)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            scores.append(accuracy_score(y_val, pred))

        return np.mean(scores)

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Run Bayesian optimization for all models."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if verbose:
            print("=" * 60)
            print("BAYESIAN HYPERPARAMETER OPTIMIZATION")
            print("=" * 60)

        # Optimize XGBoost
        if verbose:
            print("\n[1/4] Optimizing XGBoost...")
        study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_state))
        study_xgb.optimize(lambda trial: self._objective_xgb(trial, X, y), n_trials=self.n_trials, show_progress_bar=False)
        self.best_params['xgb'] = study_xgb.best_params
        if verbose:
            print(f"    Best XGBoost CV Score: {study_xgb.best_value:.4f}")

        # Optimize LightGBM
        if verbose:
            print("\n[2/4] Optimizing LightGBM...")
        study_lgbm = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_state))
        study_lgbm.optimize(lambda trial: self._objective_lgbm(trial, X, y), n_trials=self.n_trials, show_progress_bar=False)
        self.best_params['lgbm'] = study_lgbm.best_params
        if verbose:
            print(f"    Best LightGBM CV Score: {study_lgbm.best_value:.4f}")

        # Optimize CatBoost
        if verbose:
            print("\n[3/4] Optimizing CatBoost...")
        study_cat = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_state))
        study_cat.optimize(lambda trial: self._objective_catboost(trial, X, y), n_trials=self.n_trials, show_progress_bar=False)
        self.best_params['catboost'] = study_cat.best_params
        if verbose:
            print(f"    Best CatBoost CV Score: {study_cat.best_value:.4f}")

        # Optimize Random Forest
        if verbose:
            print("\n[4/4] Optimizing Random Forest...")
        study_rf = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_state))
        study_rf.optimize(lambda trial: self._objective_rf(trial, X, y), n_trials=self.n_trials, show_progress_bar=False)
        self.best_params['rf'] = study_rf.best_params
        if verbose:
            print(f"    Best Random Forest CV Score: {study_rf.best_value:.4f}")

        # Store best CV scores for weighting
        self.model_weights = {
            'xgb': study_xgb.best_value,
            'lgbm': study_lgbm.best_value,
            'catboost': study_cat.best_value,
            'rf': study_rf.best_value
        }

        if verbose:
            print("\n" + "=" * 60)
            print("Optimization Complete!")
            print("=" * 60)

    def build_models(self):
        """Build models with optimized hyperparameters."""
        # XGBoost
        xgb_params = self.best_params.get('xgb', {})
        xgb_params['random_state'] = self.random_state
        xgb_params['use_label_encoder'] = False
        xgb_params['eval_metric'] = 'logloss'
        self.models['xgb'] = XGBClassifier(**xgb_params)

        # LightGBM
        lgbm_params = self.best_params.get('lgbm', {})
        lgbm_params['random_state'] = self.random_state
        lgbm_params['verbosity'] = -1
        self.models['lgbm'] = LGBMClassifier(**lgbm_params)

        # CatBoost
        cat_params = self.best_params.get('catboost', {})
        cat_params['random_seed'] = self.random_state
        cat_params['verbose'] = False
        self.models['catboost'] = CatBoostClassifier(**cat_params)

        # Random Forest
        rf_params = self.best_params.get('rf', {})
        rf_params['random_state'] = self.random_state
        rf_params['n_jobs'] = 1
        self.models['rf'] = RandomForestClassifier(**rf_params)

        # Extra Trees (uses RF params as base)
        et_params = rf_params.copy()
        et_params['n_jobs'] = 1
        self.models['extratrees'] = ExtraTreesClassifier(**et_params)

        # Gradient Boosting
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=self.random_state
        )

        # Logistic Regression (for diversity)
        self.models['lr'] = LogisticRegression(
            C=1.0, max_iter=1000, random_state=self.random_state
        )

        # SVM (calibrated for probability)
        self.models['svm'] = CalibratedClassifierCV(
            SVC(kernel='rbf', C=1.0, gamma='scale', random_state=self.random_state),
            cv=5
        )

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Fit all models."""
        # Scale features for linear models
        X_scaled = self.scaler.fit_transform(X)

        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING ENSEMBLE MODELS")
            print("=" * 60)

        for name, model in self.models.items():
            if verbose:
                print(f"\nTraining {name}...")

            # Use scaled features for linear models
            if name in ['lr', 'svm']:
                model.fit(X_scaled, y)
            else:
                model.fit(X, y)

            if verbose:
                print(f"    {name} trained successfully")

    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get probability predictions from all models."""
        X_scaled = self.scaler.transform(X)
        predictions = {}

        for name, model in self.models.items():
            if name in ['lr', 'svm']:
                proba = model.predict_proba(X_scaled)[:, 1]
            else:
                proba = model.predict_proba(X)[:, 1]
            predictions[name] = proba

        return predictions

    def weighted_ensemble_predict(self, X: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Get weighted ensemble predictions."""
        all_proba = self.predict_proba(X)

        # Normalize weights
        total_weight = sum(self.model_weights.values())
        normalized_weights = {k: v / total_weight for k, v in self.model_weights.items()}

        # Weighted average of probabilities
        ensemble_proba = np.zeros(X.shape[0])
        for name, proba in all_proba.items():
            weight = normalized_weights.get(name, 1.0 / len(all_proba))
            ensemble_proba += weight * proba

        # Binary predictions
        predictions = (ensemble_proba >= threshold).astype(int)

        return predictions, ensemble_proba


class StackingEnsemble:
    """
    Two-level stacking ensemble with cross-validation.
    """

    def __init__(self, base_models: Dict, random_state: int = 42):
        self.base_models = base_models
        self.random_state = random_state
        self.meta_model = LogisticRegression(C=0.5, max_iter=1000, random_state=random_state)
        self.scaler = StandardScaler()
        self.meta_features_scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Fit stacking ensemble using out-of-fold predictions."""
        n_samples = X.shape[0]
        n_models = len(self.base_models)

        # Generate out-of-fold predictions for meta-features
        meta_features = np.zeros((n_samples, n_models))

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        X_scaled = self.scaler.fit_transform(X)

        if verbose:
            print("\n" + "=" * 60)
            print("BUILDING STACKING ENSEMBLE")
            print("=" * 60)
            print("\nGenerating out-of-fold predictions...")

        for i, (name, model) in enumerate(self.base_models.items()):
            if verbose:
                print(f"  Processing {name}...")

            oof_preds = np.zeros(n_samples)

            for train_idx, val_idx in cv.split(X, y):
                if name in ['lr', 'svm']:
                    X_train = X_scaled[train_idx]
                    X_val = X_scaled[val_idx]
                else:
                    X_train = X[train_idx]
                    X_val = X[val_idx]

                y_train = y[train_idx]

                # Clone model for each fold
                import copy
                fold_model = copy.deepcopy(model)
                fold_model.fit(X_train, y_train)

                oof_preds[val_idx] = fold_model.predict_proba(X_val)[:, 1]

            meta_features[:, i] = oof_preds

        # Fit meta-model on out-of-fold predictions
        if verbose:
            print("\nFitting meta-learner...")

        meta_features_scaled = self.meta_features_scaler.fit_transform(meta_features)
        self.meta_model.fit(meta_features_scaled, y)

        # Refit base models on full training data
        if verbose:
            print("Refitting base models on full data...")

        for name, model in self.base_models.items():
            if name in ['lr', 'svm']:
                model.fit(X_scaled, y)
            else:
                model.fit(X, y)

        if verbose:
            print("Stacking ensemble complete!")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get stacking ensemble probability predictions."""
        X_scaled = self.scaler.transform(X)

        meta_features = np.zeros((X.shape[0], len(self.base_models)))

        for i, (name, model) in enumerate(self.base_models.items()):
            if name in ['lr', 'svm']:
                meta_features[:, i] = model.predict_proba(X_scaled)[:, 1]
            else:
                meta_features[:, i] = model.predict_proba(X)[:, 1]

        meta_features_scaled = self.meta_features_scaler.transform(meta_features)
        return self.meta_model.predict_proba(meta_features_scaled)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Find optimal probability threshold for classification."""
    best_threshold = 0.5
    best_score = 0

    for threshold in np.arange(0.3, 0.7, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        score = accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold


def cross_validate_model(model, X: np.ndarray, y: np.ndarray,
                         n_splits: int = 10, n_repeats: int = 5,
                         random_state: int = 42) -> Dict:
    """Perform repeated stratified cross-validation."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        import copy
        fold_model = copy.deepcopy(model)
        fold_model.fit(X_train, y_train)
        pred = fold_model.predict(X_val)
        scores.append(accuracy_score(y_val, pred))

    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'scores': scores
    }


def main():
    """Main execution function."""
    print("=" * 70)
    print("ADVANCED TITANIC SURVIVAL PREDICTION")
    print("Target: 80-85% Accuracy on Kaggle Leaderboard")
    print("=" * 70)

    # 1. LOAD DATA
    print("\n[STEP 1] Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print(f"    Training samples: {len(train_df)}")
    print(f"    Test samples: {len(test_df)}")

    # Store passenger IDs for submission
    test_ids = test_df['PassengerId'].values

    # 2. FEATURE ENGINEERING
    print("\n[STEP 2] Advanced Feature Engineering...")
    feature_engineer = TitanicFeatureEngineer()
    feature_engineer.fit(train_df)

    train_processed = feature_engineer.transform(train_df, is_train=True)
    test_processed = feature_engineer.transform(test_df, is_train=False)

    feature_cols = feature_engineer.get_feature_columns()
    print(f"    Features engineered: {len(feature_cols)}")

    X = train_processed[feature_cols].values
    y = train_df['Survived'].values
    X_test = test_processed[feature_cols].values

    print(f"    Feature matrix shape: {X.shape}")

    # 3. PROBABILISTIC WCG
    print("\n[STEP 3] Building Probabilistic WCG Model...")
    wcg = ProbabilisticWCG(threshold_confidence=0.8)
    wcg.fit(train_df)

    wcg_predictions, wcg_confidences = wcg.get_wcg_predictions(test_df)
    deterministic_cases = np.sum(~np.isnan(wcg_predictions) & (wcg_confidences > 0.5))
    print(f"    Deterministic WCG cases: {deterministic_cases}")

    # 4. HYPERPARAMETER OPTIMIZATION
    print("\n[STEP 4] Bayesian Hyperparameter Optimization...")
    ensemble = OptimizedEnsemble(n_trials=20, cv_folds=5, random_state=RANDOM_STATE)
    ensemble.optimize_hyperparameters(X, y, verbose=True)

    # 5. BUILD AND TRAIN MODELS
    print("\n[STEP 5] Building Optimized Models...")
    ensemble.build_models()
    ensemble.fit(X, y, verbose=True)

    # 6. CROSS-VALIDATION
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}

    for name, model in ensemble.models.items():
        scores = []
        X_data = ensemble.scaler.transform(X) if name in ['lr', 'svm'] else X

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X_data[train_idx], X_data[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            import copy
            fold_model = copy.deepcopy(model)
            fold_model.fit(X_train, y_train)
            pred = fold_model.predict(X_val)
            scores.append(accuracy_score(y_val, pred))

        cv_scores[name] = np.mean(scores)
        print(f"  {name:12s}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    # 7. STACKING ENSEMBLE
    print("\n[STEP 6] Building Stacking Ensemble...")
    stacking = StackingEnsemble(
        base_models={
            'xgb': ensemble.models['xgb'],
            'lgbm': ensemble.models['lgbm'],
            'catboost': ensemble.models['catboost'],
            'rf': ensemble.models['rf']
        },
        random_state=RANDOM_STATE
    )
    stacking.fit(X, y, verbose=True)

    # Cross-validate stacking
    stacking_cv_scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        import copy
        fold_stacking = copy.deepcopy(stacking)
        fold_stacking.fit(X_train, y_train, verbose=False)
        pred = fold_stacking.predict(X_val)
        stacking_cv_scores.append(accuracy_score(y_val, pred))

    print(f"\n  Stacking Ensemble CV: {np.mean(stacking_cv_scores):.4f} (+/- {np.std(stacking_cv_scores):.4f})")

    # 8. WEIGHTED ENSEMBLE PREDICTIONS
    print("\n[STEP 7] Generating Predictions...")

    # Get base ensemble predictions
    _, ensemble_proba = ensemble.weighted_ensemble_predict(X_test)

    # Get stacking predictions
    stacking_proba = stacking.predict_proba(X_test)

    # Blend stacking with weighted ensemble
    final_proba = 0.5 * ensemble_proba + 0.5 * stacking_proba

    # Apply WCG blending
    final_proba_wcg = wcg.blend_predictions(final_proba, wcg_predictions, wcg_confidences, blend_strength=0.7)

    # Find optimal threshold using training data OOF predictions
    print("\n[STEP 8] Optimizing Decision Threshold...")
    oof_proba = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, _ = y[train_idx], y[val_idx]

        import copy
        fold_stacking = copy.deepcopy(stacking)
        fold_stacking.fit(X_train, y_train, verbose=False)
        oof_proba[val_idx] = fold_stacking.predict_proba(X_val)

    optimal_threshold = find_optimal_threshold(y, oof_proba)
    print(f"    Optimal threshold: {optimal_threshold:.3f}")

    # Final predictions
    final_predictions = (final_proba_wcg >= optimal_threshold).astype(int)

    # 9. GENERATE SUBMISSION
    print("\n[STEP 9] Generating Submission File...")
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': final_predictions
    })
    submission.to_csv('submission_advanced.csv', index=False)

    # Also save with standard threshold for comparison
    submission_std = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': (final_proba_wcg >= 0.5).astype(int)
    })
    submission_std.to_csv('submission_advanced_std.csv', index=False)

    print(f"    Submission saved: submission_advanced.csv")
    print(f"    Predictions distribution: {np.sum(final_predictions == 1)} survived, {np.sum(final_predictions == 0)} died")

    # 10. SUMMARY
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Features: {len(feature_cols)}")
    print(f"  Models in ensemble: {len(ensemble.models)}")
    print(f"  Best single model CV: {max(cv_scores.values()):.4f}")
    print(f"  Stacking ensemble CV: {np.mean(stacking_cv_scores):.4f}")
    print(f"  WCG deterministic overrides: {deterministic_cases}")
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    print("\n  Expected Kaggle Score: 80-85% (based on CV performance)")
    print("=" * 70)

    return submission


if __name__ == "__main__":
    main()
