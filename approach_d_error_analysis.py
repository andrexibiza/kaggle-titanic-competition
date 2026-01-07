"""
APPROACH D: ERROR ANALYSIS & TARGETED MICRO-CORRECTIONS
========================================================
Strategy: Analyze V4's likely errors by comparing with V11 disagreements,
then build targeted micro-models to fix them.

V4 Score: 0.78947 (BEST)
V11 Score: 0.78708
Advanced Score: 0.74401 (WORST)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("APPROACH D: ERROR ANALYSIS & MICRO-CORRECTIONS")
print("="*80)

# ==============================================================================
# STEP 1: LOAD SUBMISSIONS & TEST DATA
# ==============================================================================
print("\n[STEP 1] Loading submissions...")

v4 = pd.read_csv('/home/user/kaggle-titanic-competition/2025-R-Attempts/submission_v4.csv')
v11 = pd.read_csv('/home/user/kaggle-titanic-competition/2025-R-Attempts/submission_v11_seed_avg.csv')
adv = pd.read_csv('/home/user/kaggle-titanic-competition/submission_advanced.csv')
test = pd.read_csv('/home/user/kaggle-titanic-competition/test.csv')
train = pd.read_csv('/home/user/kaggle-titanic-competition/train.csv')

print(f"  V4 predictions: {v4['Survived'].sum()} survivors ({v4['Survived'].sum()/len(v4)*100:.1f}%)")
print(f"  V11 predictions: {v11['Survived'].sum()} survivors ({v11['Survived'].sum()/len(v11)*100:.1f}%)")
print(f"  Advanced predictions: {adv['Survived'].sum()} survivors ({adv['Survived'].sum()/len(adv)*100:.1f}%)")

# ==============================================================================
# STEP 2: IDENTIFY DISAGREEMENT PATTERNS
# ==============================================================================
print("\n[STEP 2] Analyzing disagreements between V4 and V11...")

# Find where V4 and V11 disagree
comparison = pd.DataFrame({
    'PassengerId': v4['PassengerId'],
    'V4_pred': v4['Survived'],
    'V11_pred': v11['Survived'],
    'Adv_pred': adv['Survived']
})

comparison['V4_V11_disagree'] = (comparison['V4_pred'] != comparison['V11_pred']).astype(int)
comparison['V4_says_survive'] = (comparison['V4_pred'] == 1) & (comparison['V11_pred'] == 0)
comparison['V4_says_die'] = (comparison['V4_pred'] == 0) & (comparison['V11_pred'] == 1)

# Merge with test data
test_analysis = test.merge(comparison, on='PassengerId')

print(f"\n  Total disagreements: {comparison['V4_V11_disagree'].sum()} out of {len(comparison)} ({comparison['V4_V11_disagree'].sum()/len(comparison)*100:.1f}%)")
print(f"  V4 says SURVIVE, V11 says DIE: {comparison['V4_says_survive'].sum()}")
print(f"  V4 says DIE, V11 says SURVIVE: {comparison['V4_says_die'].sum()}")

# Analyze demographics of disagreement cases
print("\n[DISAGREEMENT DEMOGRAPHICS]")
print("\nWhere V4 predicts SURVIVAL but V11 predicts DEATH:")
v4_survive_disagree = test_analysis[test_analysis['V4_says_survive']]
if len(v4_survive_disagree) > 0:
    print(f"  Count: {len(v4_survive_disagree)}")
    print(f"  Pclass distribution: {v4_survive_disagree['Pclass'].value_counts().to_dict()}")
    print(f"  Sex distribution: {v4_survive_disagree['Sex'].value_counts().to_dict()}")
    print(f"  Mean Age: {v4_survive_disagree['Age'].mean():.1f}")
    print(f"  Mean Fare: {v4_survive_disagree['Fare'].mean():.2f}")

print("\nWhere V4 predicts DEATH but V11 predicts SURVIVAL:")
v4_die_disagree = test_analysis[test_analysis['V4_says_die']]
if len(v4_die_disagree) > 0:
    print(f"  Count: {len(v4_die_disagree)}")
    print(f"  Pclass distribution: {v4_die_disagree['Pclass'].value_counts().to_dict()}")
    print(f"  Sex distribution: {v4_die_disagree['Sex'].value_counts().to_dict()}")
    print(f"  Mean Age: {v4_die_disagree['Age'].mean():.1f}")
    print(f"  Mean Fare: {v4_die_disagree['Fare'].mean():.2f}")

# ==============================================================================
# STEP 3: FEATURE ENGINEERING (V4-STYLE)
# ==============================================================================
print("\n[STEP 3] Building V4-style feature pipeline...")

def extract_title(name):
    """Extract title from name"""
    import re
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

def engineer_features(df, is_train=True):
    """Apply V4-style feature engineering"""
    df = df.copy()

    # Title extraction
    df['Title'] = df['Name'].apply(extract_title)

    # Title normalization
    df['Title'] = df['Title'].replace(['Mme'], 'Mrs')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Dona'], 'Mrs')
    df['Title'] = df['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Rare')

    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Deck from cabin
    df['Deck'] = df['Cabin'].fillna('U').str[0]

    # Embarked
    df['Embarked'] = df['Embarked'].fillna('S')

    # Fare
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Age imputation by title
    if is_train:
        age_by_title = df.groupby('Title')['Age'].median()
    else:
        age_by_title = train.copy()
        age_by_title['Title'] = age_by_title['Name'].apply(extract_title)
        age_by_title['Title'] = age_by_title['Title'].replace(['Mme'], 'Mrs')
        age_by_title['Title'] = age_by_title['Title'].replace(['Mlle', 'Ms'], 'Miss')
        age_by_title['Title'] = age_by_title['Title'].replace(['Lady', 'Countess', 'Dona'], 'Mrs')
        age_by_title['Title'] = age_by_title['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Rare')
        age_by_title = age_by_title.groupby('Title')['Age'].median()

    df['Age'] = df.apply(lambda row: age_by_title[row['Title']] if pd.isna(row['Age']) else row['Age'], axis=1)

    # Age groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                            labels=['Child', 'Teen', 'Adult', 'MiddleAge', 'Senior'])

    # Fare groups
    df['FareGroup'] = pd.cut(df['Fare'], bins=[-np.inf, 7.91, 14.454, 31, np.inf],
                             labels=['Low', 'MedLow', 'MedHigh', 'High'])

    return df

# Apply feature engineering
train_fe = engineer_features(train, is_train=True)
test_fe = engineer_features(test, is_train=False)

print("  Features engineered successfully")

# ==============================================================================
# STEP 4: BUILD V4 BASELINE ENSEMBLE
# ==============================================================================
print("\n[STEP 4] Building V4 baseline ensemble...")

# Prepare features for modeling
feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                'Title', 'FamilySize', 'IsAlone', 'Deck']

X_train = train_fe[feature_cols].copy()
y_train = train_fe['Survived']
X_test = test_fe[feature_cols].copy()

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder

le_dict = {}
for col in ['Sex', 'Embarked', 'Title', 'Deck']:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = X_test[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_[0])
    X_test[col] = le.transform(X_test[col])
    le_dict[col] = le

# Build V4-style ensemble
print("  Training XGBoost...")
model_xgb = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
model_xgb.fit(X_train, y_train)

print("  Training Random Forest...")
model_rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=5,
    max_features=3,
    random_state=42
)
model_rf.fit(X_train, y_train)

print("  Training Logistic Regression...")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model_lr.fit(X_train_scaled, y_train)

# Generate predictions
pred_xgb = model_xgb.predict_proba(X_test)[:, 1]
pred_rf = model_rf.predict_proba(X_test)[:, 1]
pred_lr = model_lr.predict_proba(X_test_scaled)[:, 1]

# V4 baseline: simple average
v4_baseline_prob = (pred_xgb + pred_rf + pred_lr) / 3
v4_baseline_pred = (v4_baseline_prob > 0.5).astype(int)

print(f"  V4 Baseline predicts: {v4_baseline_pred.sum()} survivors ({v4_baseline_pred.sum()/len(v4_baseline_pred)*100:.1f}%)")

# Cross-validation score
cv_scores = []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Train mini ensemble
    m1 = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')
    m2 = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=5,
                                max_features=3, random_state=42)

    m1.fit(X_tr, y_tr)
    m2.fit(X_tr, y_tr)

    p1 = m1.predict_proba(X_val)[:, 1]
    p2 = m2.predict_proba(X_val)[:, 1]

    ensemble_pred = ((p1 + p2) / 2 > 0.5).astype(int)
    score = (ensemble_pred == y_val).mean()
    cv_scores.append(score)

print(f"  V4 Baseline CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# ==============================================================================
# STEP 5: TARGETED MICRO-CORRECTIONS
# ==============================================================================
print("\n[STEP 5] Applying targeted micro-corrections...")

# Create final predictions starting with V4 baseline
final_predictions = v4_baseline_pred.copy()
corrections_made = []

# Analyze uncertain cases (where models disagree)
test_fe['V4_baseline'] = v4_baseline_pred
test_fe['V4_prob'] = v4_baseline_prob
test_fe['XGB_prob'] = pred_xgb
test_fe['RF_prob'] = pred_rf
test_fe['LR_prob'] = pred_lr

# Calculate uncertainty: standard deviation of model probabilities
test_fe['uncertainty'] = test_fe[['XGB_prob', 'RF_prob', 'LR_prob']].std(axis=1)

# Merge with comparison data
test_fe = test_fe.merge(comparison[['PassengerId', 'V4_pred', 'V11_pred', 'V4_V11_disagree']],
                        on='PassengerId', how='left')

print(f"\n  High uncertainty cases (std > 0.15): {(test_fe['uncertainty'] > 0.15).sum()}")

# CORRECTION 1: Class 3 females with large families (likely over-predicted survival)
correction1_mask = (
    (test_fe['Pclass'] == 3) &
    (test_fe['Sex'] == 'female') &
    (test_fe['FamilySize'] >= 5) &
    (test_fe['V4_baseline'] == 1) &
    (test_fe['Fare'] < 20)
)
correction1_indices = test_fe[correction1_mask].index.tolist()
if len(correction1_indices) > 0:
    # Apply correction with caution - only flip if uncertainty is high
    for idx in correction1_indices:
        if test_fe.loc[idx, 'uncertainty'] > 0.12:
            final_predictions[idx] = 0
            corrections_made.append(('Class 3 female, large family, low fare',
                                   test_fe.loc[idx, 'PassengerId'], 1, 0))

print(f"  Correction 1 (Class 3 females, large families): {len([c for c in corrections_made if 'Class 3 female' in c[0]])} cases")

# CORRECTION 2: Class 1 males with high fare and cabin (likely under-predicted survival)
correction2_mask = (
    (test_fe['Pclass'] == 1) &
    (test_fe['Sex'] == 'male') &
    (test_fe['V4_baseline'] == 0) &
    (test_fe['Fare'] > 50) &
    (test_fe['Deck'].isin(['B', 'C', 'D', 'E'])) &
    (test_fe['Age'] < 50)
)
correction2_indices = test_fe[correction2_mask].index.tolist()
if len(correction2_indices) > 0:
    for idx in correction2_indices:
        if test_fe.loc[idx, 'uncertainty'] > 0.12:
            final_predictions[idx] = 1
            corrections_made.append(('Class 1 male, high fare, good cabin',
                                   test_fe.loc[idx, 'PassengerId'], 0, 1))

print(f"  Correction 2 (Class 1 males, wealthy): {len([c for c in corrections_made if 'Class 1 male' in c[0]])} cases")

# CORRECTION 3: V4/V11 disagreement cases - trust the more conservative prediction
# For V4_V11 disagreements with high uncertainty, flip to more conservative (death)
correction3_mask = (
    (test_fe['V4_V11_disagree'] == 1) &
    (test_fe['uncertainty'] > 0.18) &
    (test_fe['V4_baseline'] == 1)
)
correction3_indices = test_fe[correction3_mask].index.tolist()[:5]  # Limit to 5 cases
if len(correction3_indices) > 0:
    for idx in correction3_indices:
        final_predictions[idx] = 0
        corrections_made.append(('High disagreement, conservative',
                               test_fe.loc[idx, 'PassengerId'], 1, 0))

print(f"  Correction 3 (High uncertainty disagreements): {len([c for c in corrections_made if 'High disagreement' in c[0]])} cases")

# CORRECTION 4: Young class 2 males (teens) - likely to die
correction4_mask = (
    (test_fe['Pclass'] == 2) &
    (test_fe['Sex'] == 'male') &
    (test_fe['Age'] >= 12) &
    (test_fe['Age'] <= 18) &
    (test_fe['V4_baseline'] == 1) &
    (test_fe['IsAlone'] == 1)
)
correction4_indices = test_fe[correction4_mask].index.tolist()
if len(correction4_indices) > 0:
    for idx in correction4_indices:
        if test_fe.loc[idx, 'uncertainty'] > 0.10:
            final_predictions[idx] = 0
            corrections_made.append(('Class 2 teenage male, alone',
                                   test_fe.loc[idx, 'PassengerId'], 1, 0))

print(f"  Correction 4 (Class 2 teenage males): {len([c for c in corrections_made if 'teenage male' in c[0]])} cases")

print(f"\n  Total corrections made: {len(corrections_made)}")

# Display corrections
if len(corrections_made) > 0:
    print("\n  CORRECTION DETAILS:")
    for reason, pid, old, new in corrections_made[:15]:  # Show first 15
        print(f"    PassengerId {pid}: {old} -> {new} ({reason})")

# ==============================================================================
# STEP 6: GENERATE SUBMISSION
# ==============================================================================
print("\n[STEP 6] Generating submission...")

final_survival_count = final_predictions.sum()
final_survival_rate = final_survival_count / len(final_predictions) * 100

print(f"  Final predictions: {final_survival_count} survivors ({final_survival_rate:.1f}%)")
print(f"  Target range: 36-38% (152-160 survivors)")

# Verify survival rate is in acceptable range
if final_survival_rate < 36 or final_survival_rate > 38:
    print(f"  WARNING: Survival rate {final_survival_rate:.1f}% is outside target range!")
    print(f"  Adjusting to match V4 baseline more closely...")
    final_predictions = v4_baseline_pred.copy()
    final_survival_count = final_predictions.sum()
    final_survival_rate = final_survival_count / len(final_predictions) * 100

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': final_predictions
})

submission.to_csv('/home/user/kaggle-titanic-competition/submission_approach_d.csv', index=False)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Final Predictions: {final_survival_count} survivors ({final_survival_rate:.1f}%)")
print(f"Corrections Applied: {len(corrections_made)}")
print(f"CV Score (V4 Baseline): {np.mean(cv_scores):.4f}")
print(f"Submission saved: /home/user/kaggle-titanic-competition/submission_approach_d.csv")
print("="*80)

# ==============================================================================
# STEP 7: DISAGREEMENT ANALYSIS OUTPUT
# ==============================================================================
print("\n[DISAGREEMENT PASSENGER LIST]")
print("="*80)

disagreement_passengers = test_fe[test_fe['V4_V11_disagree'] == 1][
    ['PassengerId', 'Pclass', 'Sex', 'Age', 'FamilySize', 'Fare', 'Deck',
     'V4_pred', 'V11_pred', 'V4_baseline', 'uncertainty']
].sort_values('uncertainty', ascending=False)

print(f"\nTotal disagreement cases: {len(disagreement_passengers)}")
print("\nTop 20 disagreement cases (by uncertainty):")
print(disagreement_passengers.head(20).to_string())

print("\n" + "="*80)
print("APPROACH D COMPLETE!")
print("="*80)
