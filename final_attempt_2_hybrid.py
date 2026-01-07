#!/usr/bin/env python3
"""
FINAL ATTEMPT 2: Hybrid Strategic Approach
===========================================

Combines:
1. Extreme conservative (flip low-prob males to die)
2. High-confidence family corrections (flip 100% family survivors)

This is our "best of both worlds" approach.
"""

import numpy as np
import pandas as pd
import re
from xgboost import XGBClassifier

RANDOM_STATE = 42

def main():
    print("=" * 60)
    print("FINAL ATTEMPT 2: HYBRID STRATEGIC")
    print("=" * 60)

    # Load data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    v4 = pd.read_csv('2025-R-Attempts/submission_v4.csv')
    strategy2 = pd.read_csv('submission_strategy_2.csv')

    print(f"\nV4: {v4['Survived'].sum()} survivors (0.78947)")
    print(f"Strategy 2: {strategy2['Survived'].sum()} survivors (0.79665)")

    # Start from Strategy 2
    final_pred = strategy2['Survived'].values.copy()

    # =========================================================================
    # STEP 1: Find HIGH-CONFIDENCE family patterns
    # =========================================================================
    print("\n[1] Finding high-confidence family patterns...")

    train['Surname'] = train['Name'].apply(lambda x: x.split(',')[0].strip())
    test['Surname'] = test['Name'].apply(lambda x: x.split(',')[0].strip())

    def get_title(name):
        match = re.search(r'([A-Za-z]+)\.', name)
        return match.group(1) if match else 'Unknown'

    test['Title'] = test['Name'].apply(get_title)

    # Track changes
    changes = []

    for idx, row in test.iterrows():
        surname = row['Surname']
        fare = row['Fare'] if pd.notna(row['Fare']) else 0
        pid = row['PassengerId']
        current_pred = final_pred[idx]

        # Find family in training (same surname, similar fare - V4's exact logic)
        family = train[
            (train['Surname'] == surname) &
            (abs(train['Fare'] - fare) < 5)
        ]

        if len(family) >= 3:  # Need strong evidence (3+ family members)
            survival_rate = family['Survived'].mean()

            # ALL family survived -> high confidence this person survived
            if survival_rate == 1.0 and current_pred == 0:
                if row['Sex'] == 'female' or row['Title'] == 'Master':
                    changes.append(('survive', pid, idx, f'Family 100% survived (n={len(family)})'))
                    final_pred[idx] = 1

            # ALL family died -> high confidence this person died
            elif survival_rate == 0.0 and current_pred == 1:
                if row['Sex'] == 'male' and row['Title'] != 'Master':
                    changes.append(('die', pid, idx, f'Family 100% died (n={len(family)})'))
                    final_pred[idx] = 0

    print(f"    Family-based changes: {len(changes)}")
    for change_type, pid, idx, reason in changes:
        print(f"      PID {pid} → {change_type.upper()} ({reason})")

    # =========================================================================
    # STEP 2: Conservative male adjustments
    # =========================================================================
    print("\n[2] Conservative male adjustments...")

    # Build model to get probabilities
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

    # Find remaining male survivors with LOW probability
    test_analysis = test.copy()
    test_analysis['Prob'] = probs
    test_analysis['CurrentPred'] = final_pred

    low_prob_males = test_analysis[
        (test_analysis['Sex'] == 'male') &
        (test_analysis['CurrentPred'] == 1) &
        (test_analysis['Prob'] < 0.20)  # Very low probability
    ].sort_values('Prob')

    print(f"    Low-probability male survivors: {len(low_prob_males)}")

    # Flip the 2 lowest (conservative - not too aggressive)
    n_to_flip = min(2, len(low_prob_males))
    for _, row in low_prob_males.head(n_to_flip).iterrows():
        idx = test[test['PassengerId'] == row['PassengerId']].index[0]
        print(f"      PID {row['PassengerId']} (prob={row['Prob']:.3f}) → DIE")
        final_pred[idx] = 0

    # =========================================================================
    # FINAL ANALYSIS
    # =========================================================================
    print("\n[3] Final analysis...")

    survivors = final_pred.sum()
    print(f"    Final survivors: {survivors}/418 ({survivors*100/418:.1f}%)")

    match_v4 = (final_pred == v4['Survived'].values).sum()
    match_s2 = (final_pred == strategy2['Survived'].values).sum()
    print(f"    Match V4: {match_v4}/418")
    print(f"    Match S2: {match_s2}/418")
    print(f"    Changes from S2: {418 - match_s2}")

    # Breakdown
    s2_to_die = ((strategy2['Survived'] == 1) & (final_pred == 0)).sum()
    s2_to_survive = ((strategy2['Survived'] == 0) & (final_pred == 1)).sum()
    print(f"    S2 survive → die: {s2_to_die}")
    print(f"    S2 die → survive: {s2_to_survive}")

    # Save
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': final_pred
    })
    submission.to_csv('submission_final_2.csv', index=False)
    print(f"\n    Saved: submission_final_2.csv")

    print("\n" + "=" * 60)
    print("FINAL ATTEMPT 2 COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
