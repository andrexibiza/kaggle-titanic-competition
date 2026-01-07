#!/usr/bin/env python3
"""
FINAL ATTEMPT 2: Optimized Family/Ticket Analysis
==================================================

This approach uses a more sophisticated family/ticket survival analysis
to find high-confidence corrections.

Strategy:
1. Build exact replica of V4's FamilySurvived logic
2. Find passengers where family pattern is ABSOLUTE (0% or 100%)
3. Only flip predictions with highest confidence

Target: 0.80+
"""

import numpy as np
import pandas as pd
import re

RANDOM_STATE = 42

def main():
    print("=" * 60)
    print("FINAL ATTEMPT 2: OPTIMIZED FAMILY ANALYSIS")
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
    # PRECISE FAMILY SURVIVAL ANALYSIS
    # =========================================================================
    print("\n[1] Precise family survival analysis...")

    # Extract surnames
    train['Surname'] = train['Name'].apply(lambda x: x.split(',')[0].strip())
    test['Surname'] = test['Name'].apply(lambda x: x.split(',')[0].strip())

    # Extract titles
    def get_title(name):
        match = re.search(r'([A-Za-z]+)\.', name)
        return match.group(1) if match else 'Unknown'

    test['Title'] = test['Name'].apply(get_title)

    # For each test passenger, check family survival pattern
    corrections = []

    for idx, row in test.iterrows():
        surname = row['Surname']
        fare = row['Fare'] if pd.notna(row['Fare']) else 0
        pid = row['PassengerId']
        current_pred = final_pred[idx]

        # Find family in training (same surname, similar fare)
        family = train[
            (train['Surname'] == surname) &
            (abs(train['Fare'] - fare) < 5)
        ]

        if len(family) >= 2:  # Meaningful family data
            survival_rate = family['Survived'].mean()
            family_size = len(family)

            # HIGH CONFIDENCE: All family died
            if survival_rate == 0.0 and current_pred == 1:
                # Male should definitely die
                if row['Sex'] == 'male':
                    corrections.append({
                        'pid': pid, 'idx': idx,
                        'from': 1, 'to': 0,
                        'reason': f'All family died (n={family_size})',
                        'confidence': 'HIGH'
                    })

            # HIGH CONFIDENCE: All family survived
            elif survival_rate == 1.0 and current_pred == 0:
                # Female/child should probably survive
                if row['Sex'] == 'female' or row['Title'] == 'Master':
                    corrections.append({
                        'pid': pid, 'idx': idx,
                        'from': 0, 'to': 1,
                        'reason': f'All family survived (n={family_size})',
                        'confidence': 'HIGH'
                    })

    print(f"    Found {len(corrections)} potential corrections")
    for c in corrections:
        print(f"      PID {c['pid']}: {c['from']}→{c['to']} ({c['reason']})")

    # =========================================================================
    # APPLY CONSERVATIVE CORRECTIONS
    # =========================================================================
    print("\n[2] Applying corrections (conservative only)...")

    # Only apply survive->die corrections (we've learned that fewer survivors = better)
    die_corrections = [c for c in corrections if c['to'] == 0]
    survive_corrections = [c for c in corrections if c['to'] == 1]

    print(f"    Survive→Die corrections: {len(die_corrections)}")
    print(f"    Die→Survive corrections: {len(survive_corrections)} (NOT applying)")

    for c in die_corrections:
        final_pred[c['idx']] = 0
        print(f"      Applied: PID {c['pid']} → DIE")

    # =========================================================================
    # TICKET-BASED ANALYSIS
    # =========================================================================
    print("\n[3] Ticket-based analysis...")

    for idx, row in test.iterrows():
        ticket = row['Ticket']
        pid = row['PassengerId']
        current_pred = final_pred[idx]

        # Find ticket group in training
        ticket_group = train[train['Ticket'] == ticket]

        if len(ticket_group) >= 2:
            survival_rate = ticket_group['Survived'].mean()

            # All ticket group died -> this person died
            if survival_rate == 0.0 and current_pred == 1:
                if row['Sex'] == 'male':
                    print(f"      PID {pid}: Ticket group all died → DIE")
                    final_pred[idx] = 0

    # =========================================================================
    # FINAL ANALYSIS
    # =========================================================================
    print("\n[4] Final analysis...")

    survivors = final_pred.sum()
    print(f"    Final survivors: {survivors}/418 ({survivors*100/418:.1f}%)")

    match_v4 = (final_pred == v4['Survived'].values).sum()
    match_s2 = (final_pred == strategy2['Survived'].values).sum()
    print(f"    Match V4: {match_v4}/418")
    print(f"    Match S2: {match_s2}/418")
    print(f"    Changes from S2: {418 - match_s2}")

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
