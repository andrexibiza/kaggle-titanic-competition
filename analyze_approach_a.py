"""Quick analysis of Approach A results"""
import pandas as pd
import numpy as np

# Load submission
submission = pd.read_csv('/home/user/kaggle-titanic-competition/submission_approach_a.csv')

# Analyze predictions
total = len(submission)
survived = submission['Survived'].sum()
died = total - survived
survival_rate = (survived / total) * 100

print("=" * 60)
print("APPROACH A - SUBMISSION ANALYSIS")
print("=" * 60)
print(f"Total predictions:    {total}")
print(f"Predicted survived:   {survived} ({survival_rate:.2f}%)")
print(f"Predicted died:       {died} ({100-survival_rate:.2f}%)")
print()
print(f"Target survival rate: ~37%")
print(f"Actual survival rate: {survival_rate:.2f}%")
print(f"Difference:           {survival_rate - 37:.2f}%")
print("=" * 60)

# Load train data to check actual survival rate
train = pd.read_csv('/home/user/kaggle-titanic-competition/train.csv')
train_survival_rate = (train['Survived'].sum() / len(train)) * 100
print(f"\nTraining survival rate: {train_survival_rate:.2f}%")
print("=" * 60)

# Distribution check
print("\nPrediction distribution:")
print(submission['Survived'].value_counts().sort_index())
