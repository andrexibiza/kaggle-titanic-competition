"""
Detailed analysis script for Approach B results
"""
import pandas as pd
import numpy as np
from approach_b_svm_ensemble import load_data, feature_engineering, prepare_features

def analyze_features():
    """Analyze feature distributions and correlations"""
    print("="*60)
    print("FEATURE ANALYSIS")
    print("="*60)

    # Load data
    train, test = load_data()
    data = feature_engineering(train, test)
    X_train, y_train, X_test, test_df = prepare_features(data, len(train))

    print("\nFeature Statistics (Training Set):")
    print(X_train.describe().T)

    print("\n\nFeature Correlations with Survival:")
    correlations = pd.DataFrame({
        'Feature': X_train.columns,
        'Correlation': [X_train[col].corr(y_train) for col in X_train.columns]
    }).sort_values('Correlation', ascending=False)
    print(correlations.to_string(index=False))

    print("\n\nSurvival Rates by Key Features:")

    # Pclass
    print("\nBy Pclass:")
    print(train.groupby('Pclass')['Survived'].agg(['mean', 'count']))

    # Sex
    print("\nBy Sex:")
    print(train.groupby('Sex')['Survived'].agg(['mean', 'count']))

    # IsAlone (need to calculate)
    train['IsAlone'] = ((train['SibSp'] + train['Parch'] + 1) == 1).astype(int)
    print("\nBy IsAlone:")
    print(train.groupby('IsAlone')['Survived'].agg(['mean', 'count']))

    # Title
    train['Title'] = train['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    print("\nBy Title:")
    print(train.groupby('Title')['Survived'].agg(['mean', 'count']).sort_values('count', ascending=False))


def analyze_predictions():
    """Analyze prediction distributions"""
    print("\n" + "="*60)
    print("PREDICTION ANALYSIS")
    print("="*60)

    submission = pd.read_csv('/home/user/kaggle-titanic-competition/submission_approach_b.csv')
    test = pd.read_csv('/home/user/kaggle-titanic-competition/test.csv')

    # Merge to get test passenger details
    test_with_pred = test.merge(submission, on='PassengerId')

    print("\nPredicted Survival by Pclass:")
    print(test_with_pred.groupby('Pclass')['Survived'].agg(['mean', 'sum', 'count']))

    print("\nPredicted Survival by Sex:")
    print(test_with_pred.groupby('Sex')['Survived'].agg(['mean', 'sum', 'count']))

    print("\nPredicted Survival by Embarked:")
    print(test_with_pred.groupby('Embarked')['Survived'].agg(['mean', 'sum', 'count']))


if __name__ == "__main__":
    analyze_features()
    analyze_predictions()
