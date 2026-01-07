"""
Approach B: SVM-Based Ensemble for Kaggle Titanic Competition
3-model ensemble: SVM (RBF kernel) + Random Forest + Logistic Regression
Simple averaging of calibrated probabilities
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
warnings.filterwarnings('ignore')


def load_data():
    """Load train and test datasets"""
    train = pd.read_csv('/home/user/kaggle-titanic-competition/train.csv')
    test = pd.read_csv('/home/user/kaggle-titanic-competition/test.csv')
    return train, test


def extract_title(name):
    """Extract title from name"""
    title = name.split(',')[1].split('.')[0].strip()
    # Normalize rare titles
    if title in ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
        return 'Rare'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Mme':
        return 'Mrs'
    return title


def feature_engineering(train, test):
    """
    Create features for the model (keeping it simple: ~10-12 features)

    Features:
    1. Pclass
    2. Sex (encoded)
    3. Age (imputed by Title median)
    4. Fare (standardized)
    5. FamilySize = SibSp + Parch + 1
    6. Title (Mr=0, Miss=1, Mrs=2, Master=3, Rare=4)
    7. IsAlone = (FamilySize == 1)
    8. Embarked (encoded)
    9. FamilySurvived (default 0.5)
    10. TicketSurvived (default 0.5)
    """
    # Combine datasets for consistent feature engineering
    data = pd.concat([train, test], sort=False).reset_index(drop=True)

    # 1. Pclass - already numeric, keep as is

    # 2. Sex encoding
    data['Sex'] = data['Sex'].map({'female': 1, 'male': 0})

    # 3. Extract Title
    data['Title'] = data['Name'].apply(extract_title)

    # Impute Age by Title median
    data['Age'] = data.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    # Fill any remaining NaNs with overall median
    data['Age'] = data['Age'].fillna(data['Age'].median())

    # 4. Fare - fill missing values
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

    # 5. FamilySize
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    # 6. Title encoding (Mr=0, Miss=1, Mrs=2, Master=3, Rare=4)
    title_mapping = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)

    # 7. IsAlone
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

    # 8. Embarked encoding
    data['Embarked'] = data['Embarked'].fillna('S')
    embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
    data['Embarked'] = data['Embarked'].map(embarked_mapping)

    # 9. FamilySurvived - Calculate survival rate by surname (from training data only)
    data['Surname'] = data['Name'].apply(lambda x: x.split(',')[0].strip())

    # Calculate family survival rate from training data
    train_data = data[:len(train)]
    surname_survival = train_data.groupby('Surname')['Survived'].mean().to_dict()

    # Map to all data, default to 0.5
    data['FamilySurvived'] = data['Surname'].map(surname_survival).fillna(0.5)

    # 10. TicketSurvived - Calculate survival rate by ticket (from training data only)
    ticket_survival = train_data.groupby('Ticket')['Survived'].mean().to_dict()

    # Map to all data, default to 0.5
    data['TicketSurvived'] = data['Ticket'].map(ticket_survival).fillna(0.5)

    return data


def prepare_features(data, train_len):
    """Prepare features for modeling"""
    feature_columns = [
        'Pclass', 'Sex', 'Age', 'Fare', 'FamilySize',
        'Title', 'IsAlone', 'Embarked', 'FamilySurvived', 'TicketSurvived'
    ]

    # Split back into train and test
    train_df = data[:train_len].copy()
    test_df = data[train_len:].copy()

    X_train = train_df[feature_columns]
    y_train = train_df['Survived']
    X_test = test_df[feature_columns]

    return X_train, y_train, X_test, test_df


def train_ensemble(X_train, y_train, X_test):
    """
    Train 3-model ensemble:
    - SVM (RBF kernel) with calibrated probabilities
    - Random Forest
    - Logistic Regression

    Returns: ensemble predictions and individual model predictions
    """
    print("\n" + "="*60)
    print("TRAINING SVM-BASED ENSEMBLE")
    print("="*60)

    # StandardScaler for SVM and Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model 1: SVM with RBF kernel (calibrated for probabilities)
    print("\n1. Training SVM (RBF kernel) with calibration...")
    svm_base = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_calibrated = CalibratedClassifierCV(svm_base, cv=5)
    svm_calibrated.fit(X_train_scaled, y_train)
    svm_proba = svm_calibrated.predict_proba(X_test_scaled)[:, 1]
    print(f"   SVM trained and calibrated")

    # Model 2: Random Forest
    print("\n2. Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)  # RF doesn't need scaling
    rf_proba = rf.predict_proba(X_test)[:, 1]
    print(f"   Random Forest trained")

    # Model 3: Logistic Regression
    print("\n3. Training Logistic Regression...")
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    print(f"   Logistic Regression trained")

    # Ensemble: Simple averaging
    print("\n4. Creating ensemble predictions (simple averaging)...")
    ensemble_proba = (svm_proba + rf_proba + lr_proba) / 3
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)

    print(f"   Ensemble predictions created")

    return ensemble_pred, ensemble_proba, {
        'svm': svm_calibrated,
        'rf': rf,
        'lr': lr,
        'scaler': scaler,
        'svm_proba': svm_proba,
        'rf_proba': rf_proba,
        'lr_proba': lr_proba
    }


def cross_validate_ensemble(X_train, y_train, n_folds=10):
    """
    Perform 10-fold cross-validation for the ensemble
    """
    print("\n" + "="*60)
    print("10-FOLD CROSS-VALIDATION")
    print("="*60)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []

    fold_num = 1
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Scale data
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        # SVM
        svm_base = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm_calibrated = CalibratedClassifierCV(svm_base, cv=5)
        svm_calibrated.fit(X_tr_scaled, y_tr)
        svm_proba = svm_calibrated.predict_proba(X_val_scaled)[:, 1]

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        rf.fit(X_tr, y_tr)
        rf_proba = rf.predict_proba(X_val)[:, 1]

        # Logistic Regression
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(X_tr_scaled, y_tr)
        lr_proba = lr.predict_proba(X_val_scaled)[:, 1]

        # Ensemble
        ensemble_proba = (svm_proba + rf_proba + lr_proba) / 3
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)

        # Calculate accuracy
        from sklearn.metrics import accuracy_score
        score = accuracy_score(y_val, ensemble_pred)
        cv_scores.append(score)

        print(f"Fold {fold_num:2d}: {score:.4f}")
        fold_num += 1

    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)

    print(f"\n{'='*60}")
    print(f"Mean CV Score: {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"{'='*60}")

    return mean_score, std_score, cv_scores


def analyze_model_performance(models_dict, X_train, y_train):
    """Analyze individual model performance"""
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL PERFORMANCE (on full training set)")
    print("="*60)

    from sklearn.metrics import accuracy_score

    scaler = models_dict['scaler']
    X_train_scaled = scaler.transform(X_train)

    # SVM
    svm_pred = models_dict['svm'].predict(X_train_scaled)
    svm_acc = accuracy_score(y_train, svm_pred)
    print(f"\nSVM (RBF + Calibrated):    {svm_acc:.4f}")

    # Random Forest
    rf_pred = models_dict['rf'].predict(X_train)
    rf_acc = accuracy_score(y_train, rf_pred)
    print(f"Random Forest:             {rf_acc:.4f}")

    # Logistic Regression
    lr_pred = models_dict['lr'].predict(X_train_scaled)
    lr_acc = accuracy_score(y_train, lr_pred)
    print(f"Logistic Regression:       {lr_acc:.4f}")

    # Ensemble on training data
    ensemble_proba = (models_dict['svm'].predict_proba(X_train_scaled)[:, 1] +
                     models_dict['rf'].predict_proba(X_train)[:, 1] +
                     models_dict['lr'].predict_proba(X_train_scaled)[:, 1]) / 3
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    ensemble_acc = accuracy_score(y_train, ensemble_pred)
    print(f"Ensemble (Average):        {ensemble_acc:.4f}")

    return {
        'svm_acc': svm_acc,
        'rf_acc': rf_acc,
        'lr_acc': lr_acc,
        'ensemble_acc': ensemble_acc
    }


def main():
    """Main execution function"""
    print("="*60)
    print("APPROACH B: SVM-BASED ENSEMBLE")
    print("="*60)

    # Load data
    print("\nLoading data...")
    train, test = load_data()
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    # Feature engineering
    print("\nPerforming feature engineering...")
    data = feature_engineering(train, test)
    print("Features created: 10 features")

    # Prepare features
    X_train, y_train, X_test, test_df = prepare_features(data, len(train))
    print(f"\nFeature columns: {list(X_train.columns)}")
    print(f"Number of features: {len(X_train.columns)}")

    # Cross-validation
    mean_cv_score, std_cv_score, cv_scores = cross_validate_ensemble(X_train, y_train, n_folds=10)

    # Train final ensemble on full training data
    ensemble_pred, ensemble_proba, models_dict = train_ensemble(X_train, y_train, X_test)

    # Analyze model performance
    perf_metrics = analyze_model_performance(models_dict, X_train, y_train)

    # Create submission
    print("\n" + "="*60)
    print("CREATING SUBMISSION")
    print("="*60)

    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'].astype(int),
        'Survived': ensemble_pred
    })

    # Verify submission
    expected_rows = 418
    if len(submission) != expected_rows:
        raise ValueError(f"Submission has {len(submission)} rows, expected {expected_rows}!")

    # Calculate predicted survival rate
    survival_rate = submission['Survived'].mean() * 100
    survival_count = submission['Survived'].sum()

    print(f"\nSubmission shape: {submission.shape}")
    print(f"Predicted survivors: {survival_count} / {len(submission)}")
    print(f"Predicted survival rate: {survival_rate:.2f}%")

    # Check if survival rate is in target range (36-38%)
    if 36 <= survival_rate <= 38:
        print(f"✓ Survival rate within target range (36-38%)")
    else:
        print(f"⚠ Warning: Survival rate {survival_rate:.2f}% outside target range (36-38%)")

    # Save submission
    output_file = '/home/user/kaggle-titanic-competition/submission_approach_b.csv'
    submission.to_csv(output_file, index=False)
    print(f"\nSubmission saved to: {output_file}")

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\n10-Fold CV Score:          {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")
    print(f"Predicted Survivors:       {survival_count} / {len(submission)}")
    print(f"Predicted Survival Rate:   {survival_rate:.2f}%")
    print(f"\nTraining Set Performance:")
    print(f"  - SVM (RBF):             {perf_metrics['svm_acc']:.4f}")
    print(f"  - Random Forest:         {perf_metrics['rf_acc']:.4f}")
    print(f"  - Logistic Regression:   {perf_metrics['lr_acc']:.4f}")
    print(f"  - Ensemble:              {perf_metrics['ensemble_acc']:.4f}")
    print("="*60)

    return {
        'cv_score': mean_cv_score,
        'cv_std': std_cv_score,
        'survival_count': survival_count,
        'survival_rate': survival_rate,
        'performance': perf_metrics
    }


if __name__ == "__main__":
    results = main()
