#!/usr/bin/env python3
"""
ULTIMATE WCG HYBRID SUBMISSION
==============================

Based on Chris Deotte's 0.84688 methodology:
1. Train XGBoost with conservative hyperparameters
2. Apply WCG overrides for Women and Boys ONLY
3. Use Ticket-based groups (more precise than Surname)

Key insight: Families lived or died TOGETHER
- If 100% of women/boys in a family survived → all survive
- If 0% of women/boys in a family survived → all die
"""

import numpy as np
import pandas as pd
import re
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42

def get_title(name):
    match = re.search(r' ([A-Za-z]+)\.', name)
    return match.group(1) if match else 'Unknown'

def main():
    print("=" * 70)
    print("ULTIMATE WCG HYBRID SUBMISSION")
    print("=" * 70)

    # Load data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # Combine for processing
    full = pd.concat([train.assign(is_train=1), test.assign(is_train=0, Survived=np.nan)], ignore_index=True)

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    print("\n[1] Feature Engineering...")

    # Title
    full['Title'] = full['Name'].apply(get_title)
    title_map = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Lady': 'Rare',
        'Sir': 'Rare', 'Capt': 'Rare', 'Countess': 'Rare', 'Don': 'Rare',
        'Jonkheer': 'Rare', 'Dona': 'Rare'
    }
    full['Title'] = full['Title'].map(lambda x: title_map.get(x, 'Rare'))

    # Surname
    full['Surname'] = full['Name'].apply(lambda x: x.split(',')[0])

    # Age imputation by title
    age_median = full.groupby('Title')['Age'].transform('median')
    full['Age'] = full['Age'].fillna(age_median)
    full['Age'] = full['Age'].fillna(full['Age'].median())

    # Fare
    full['Fare'] = full['Fare'].fillna(full['Fare'].median())

    # Embarked
    full['Embarked'] = full['Embarked'].fillna('S')

    # Family
    full['FamilySize'] = full['SibSp'] + full['Parch'] + 1
    full['IsAlone'] = (full['FamilySize'] == 1).astype(int)

    # Encodings
    full['Sex_Enc'] = (full['Sex'] == 'male').astype(int)
    full['Embarked_Enc'] = full['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    full['Title_Enc'] = full['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})

    # Is Woman or Boy (for WCG)
    full['IsWomanOrBoy'] = ((full['Title'] == 'Master') | (full['Sex'] == 'female'))

    # =========================================================================
    # TRAIN ENSEMBLE (Conservative)
    # =========================================================================
    print("\n[2] Training Conservative Ensemble...")

    features = ['Pclass', 'Sex_Enc', 'Age', 'Fare', 'FamilySize', 'IsAlone',
                'Embarked_Enc', 'Title_Enc', 'SibSp', 'Parch']

    train_mask = full['is_train'] == 1
    X_train = full.loc[train_mask, features].values
    y_train = full.loc[train_mask, 'Survived'].values
    X_test = full.loc[~train_mask, features].values

    # XGBoost - Conservative
    model_xgb = XGBClassifier(
        n_estimators=100,
        max_depth=3,  # SHALLOW
        learning_rate=0.1,
        subsample=0.8,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )

    # Random Forest - Conservative
    model_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    )

    # Logistic Regression
    model_lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    model_xgb.fit(X_train, y_train)
    model_rf.fit(X_train, y_train)
    model_lr.fit(X_train, y_train)

    # Ensemble probabilities
    prob_xgb = model_xgb.predict_proba(X_test)[:, 1]
    prob_rf = model_rf.predict_proba(X_test)[:, 1]
    prob_lr = model_lr.predict_proba(X_test)[:, 1]

    prob_ensemble = (prob_xgb + prob_rf + prob_lr) / 3
    pred_ensemble = (prob_ensemble > 0.5).astype(int)

    print(f"    Base ensemble: {pred_ensemble.sum()} survivors")

    # =========================================================================
    # WCG LOGIC (The Secret Sauce)
    # =========================================================================
    print("\n[3] Applying WCG Override Logic...")

    train_df = full[full['is_train'] == 1].copy()
    test_df = full[full['is_train'] == 0].copy()
    test_df['BasePred'] = pred_ensemble

    # ----- GROUP IDENTIFICATION -----
    # Method 1: Ticket-based groups (strongest signal)
    # Method 2: Surname + Fare proximity (backup)

    # For each ticket, calculate the survival rate of women/boys in training
    ticket_wcg_stats = train_df[train_df['IsWomanOrBoy']].groupby('Ticket').agg({
        'Survived': ['count', 'mean', 'sum']
    })
    ticket_wcg_stats.columns = ['WCG_Count', 'WCG_Rate', 'WCG_Survived']
    ticket_wcg_stats = ticket_wcg_stats.reset_index()

    # Identify "dead tickets" and "living tickets"
    dead_tickets = set(ticket_wcg_stats[(ticket_wcg_stats['WCG_Rate'] == 0.0) &
                                         (ticket_wcg_stats['WCG_Count'] > 0)]['Ticket'])
    living_tickets = set(ticket_wcg_stats[(ticket_wcg_stats['WCG_Rate'] == 1.0) &
                                           (ticket_wcg_stats['WCG_Count'] > 0)]['Ticket'])

    # Same for surnames (with fare proximity)
    def get_surname_group_fate(surname, fare, train_df):
        """Check fate of women/boys with same surname and similar fare."""
        family = train_df[(train_df['Surname'] == surname) &
                          (train_df['IsWomanOrBoy']) &
                          (abs(train_df['Fare'] - fare) < 5)]
        if len(family) == 0:
            return None
        return family['Survived'].mean()

    print(f"    Found {len(dead_tickets)} dead tickets, {len(living_tickets)} living tickets")

    # ----- APPLY OVERRIDES -----
    final_pred = pred_ensemble.copy()
    overrides = []

    for idx, row in test_df.iterrows():
        test_idx = idx - len(train_df)  # Adjust index for prediction array

        # Only apply WCG to women and boys
        if not row['IsWomanOrBoy']:
            continue

        original_pred = final_pred[test_idx]
        new_pred = original_pred
        override_reason = None

        # Check Ticket first (strongest signal)
        if row['Ticket'] in dead_tickets:
            new_pred = 0
            override_reason = "Dead Ticket"
        elif row['Ticket'] in living_tickets:
            new_pred = 1
            override_reason = "Living Ticket"
        else:
            # Fallback to Surname + Fare
            surname_fate = get_surname_group_fate(row['Surname'], row['Fare'], train_df)
            if surname_fate is not None:
                if surname_fate == 0.0:
                    new_pred = 0
                    override_reason = "Dead Surname"
                elif surname_fate == 1.0:
                    new_pred = 1
                    override_reason = "Living Surname"

        if new_pred != original_pred:
            final_pred[test_idx] = new_pred
            overrides.append({
                'PassengerId': row['PassengerId'],
                'Name': row['Name'][:35],
                'Sex': row['Sex'],
                'Title': row['Title'],
                'From': original_pred,
                'To': new_pred,
                'Reason': override_reason
            })

    print(f"\n    WCG Overrides Applied: {len(overrides)}")

    # Show overrides
    survive_to_die = [o for o in overrides if o['From'] == 1]
    die_to_survive = [o for o in overrides if o['From'] == 0]

    print(f"\n    SURVIVE → DIE ({len(survive_to_die)}):")
    for o in survive_to_die[:10]:
        print(f"      PID {o['PassengerId']:4d}: {o['Sex']:6s} {o['Title']:6s} | {o['Reason']}")
    if len(survive_to_die) > 10:
        print(f"      ... and {len(survive_to_die) - 10} more")

    print(f"\n    DIE → SURVIVE ({len(die_to_survive)}):")
    for o in die_to_survive[:10]:
        print(f"      PID {o['PassengerId']:4d}: {o['Sex']:6s} {o['Title']:6s} | {o['Reason']}")
    if len(die_to_survive) > 10:
        print(f"      ... and {len(die_to_survive) - 10} more")

    # =========================================================================
    # CONSERVATIVE ADJUSTMENT (Our proven strategy)
    # =========================================================================
    print("\n[4] Applying Conservative Male Adjustment...")

    # Check remaining male survivors with low probability
    test_df['FinalPred'] = final_pred
    test_df['Prob'] = prob_ensemble

    male_survivors = test_df[(test_df['FinalPred'] == 1) & (test_df['Sex'] == 'male')]
    male_survivors_sorted = male_survivors.sort_values('Prob')

    print(f"    Remaining male survivors: {len(male_survivors)}")

    # Conservative: flip lowest probability adult males (not Masters/boys)
    adult_male_survivors = male_survivors_sorted[male_survivors_sorted['Title'] != 'Master']

    print(f"    Adult male survivors: {len(adult_male_survivors)}")
    if len(adult_male_survivors) > 0:
        print("    Lowest probability adult males:")
        for _, row in adult_male_survivors.head(5).iterrows():
            test_idx = row.name - len(train_df)
            print(f"      PID {row['PassengerId']:4d}: prob={row['Prob']:.3f}, Title={row['Title']}")

    # =========================================================================
    # FINAL STATS
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    survivors = final_pred.sum()
    male_survivors_final = final_pred[test_df['Sex'] == 'male'].sum()
    female_survivors_final = final_pred[test_df['Sex'] == 'female'].sum()

    print(f"\n    Total Survivors: {survivors}/418 ({survivors/418*100:.1f}%)")
    print(f"    Male Survivors: {male_survivors_final}")
    print(f"    Female Survivors: {female_survivors_final}")

    # Compare with V4
    v4 = pd.read_csv('2025-R-Attempts/submission_v4.csv')
    match_v4 = (final_pred == v4['Survived'].values).sum()
    print(f"\n    Match V4 (0.789): {match_v4}/418 ({match_v4/418*100:.1f}%)")

    # Compare with Final2
    final2 = pd.read_csv('submission_final_2.csv')
    match_f2 = (final_pred == final2['Survived'].values).sum()
    print(f"    Match Final2 (0.801): {match_f2}/418 ({match_f2/418*100:.1f}%)")

    # Save
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': final_pred.astype(int)
    })
    submission.to_csv('submission_wcg_hybrid.csv', index=False)
    print(f"\n    Saved: submission_wcg_hybrid.csv")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
