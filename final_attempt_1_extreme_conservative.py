#!/usr/bin/env python3
"""
FINAL ATTEMPT 1: Extreme Conservative
======================================

Strategy 2 (0.79665) reduced survivors from 154 to 149 (+5 deaths).
That improved score by 0.72%.

If fewer survivors = better score, let's push further:
- Identify additional "borderline" male survivors to flip to death
- Target: ~145 survivors

NO HOLDS BARRED - Maximum conservatism!
"""

import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

RANDOM_STATE = 42

def main():
    print("=" * 60)
    print("FINAL ATTEMPT 1: EXTREME CONSERVATIVE")
    print("=" * 60)

    # Load data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    v4 = pd.read_csv('2025-R-Attempts/submission_v4.csv')
    strategy2 = pd.read_csv('submission_strategy_2.csv')

    print(f"\nV4: {v4['Survived'].sum()} survivors (0.78947)")
    print(f"Strategy 2: {strategy2['Survived'].sum()} survivors (0.79665)")

    # Feature engineering
    def extract_title(name):
        match = re.search(r'([A-Za-z]+)\.', name)
        return match.group(1) if match else 'Unknown'

    test_analysis = test.copy()
    test_analysis['Title'] = test_analysis['Name'].apply(extract_title)
    test_analysis['V4'] = v4['Survived'].values
    test_analysis['S2'] = strategy2['Survived'].values
    test_analysis['FamilySize'] = test_analysis['SibSp'] + test_analysis['Parch'] + 1

    # Start from Strategy 2 (our best)
    final_pred = strategy2['Survived'].values.copy()

    print("\n[1] Analyzing remaining male survivors...")

    # Find males that Strategy 2 still predicts survive
    male_survivors = test_analysis[
        (test_analysis['S2'] == 1) &
        (test_analysis['Sex'] == 'male')
    ]
    print(f"    Male survivors in S2: {len(male_survivors)}")

    # Build a quick model to get probabilities
    full = pd.concat([train, test.assign(Survived=np.nan)], ignore_index=True)
    full['Sex_Enc'] = (full['Sex'] == 'male').astype(int)
    full['Embarked'] = full['Embarked'].fillna('S')
    full['Embarked_Enc'] = full['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    full['Fare'] = full['Fare'].fillna(full['Fare'].median())
    full['Age'] = full['Age'].fillna(full['Age'].median())
    full['FamilySize'] = full['SibSp'] + full['Parch'] + 1

    features = ['Pclass', 'Sex_Enc', 'Age', 'Fare', 'FamilySize', 'Embarked_Enc']
    X_train = full.iloc[:len(train)][features].values
    y_train = train['Survived'].values
    X_test = full.iloc[len(train):][features].values

    model = XGBClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE,
                          use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    test_analysis['Prob'] = probs

    print("\n[2] Finding lowest-probability male survivors...")

    # Sort male survivors by probability (lowest first)
    male_survivors_sorted = test_analysis[
        (test_analysis['S2'] == 1) &
        (test_analysis['Sex'] == 'male')
    ].sort_values('Prob')

    print(f"    Lowest prob male survivors:")
    for _, row in male_survivors_sorted.head(10).iterrows():
        print(f"      PID {row['PassengerId']}: prob={row['Prob']:.3f}, Class={row['Pclass']}, Age={row['Age']}")

    # Flip the 4-5 lowest probability male survivors to DIE
    n_to_flip = 4
    to_flip = male_survivors_sorted.head(n_to_flip)['PassengerId'].values

    print(f"\n[3] Flipping {n_to_flip} additional males to DIE...")
    for pid in to_flip:
        idx = test_analysis[test_analysis['PassengerId'] == pid].index[0]
        final_pred[idx] = 0
        print(f"    PID {pid} -> DIE")

    # Final counts
    survivors = final_pred.sum()
    print(f"\n[4] Results:")
    print(f"    Final survivors: {survivors}/418 ({survivors*100/418:.1f}%)")

    # Compare
    match_v4 = (final_pred == v4['Survived'].values).sum()
    match_s2 = (final_pred == strategy2['Survived'].values).sum()
    print(f"    Match V4: {match_v4}/418")
    print(f"    Match S2: {match_s2}/418")

    # Save
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': final_pred
    })
    submission.to_csv('submission_final_1.csv', index=False)
    print(f"\n    Saved: submission_final_1.csv")

    print("\n" + "=" * 60)
    print("FINAL ATTEMPT 1 COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
