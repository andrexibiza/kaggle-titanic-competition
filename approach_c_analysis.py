#!/usr/bin/env python3
"""
Detailed analysis of Approach C seed ensemble variance
"""

import numpy as np
import pandas as pd
from approach_c_seed_ensemble import (
    load_and_prepare_data, train_seed_ensemble,
    predict_seed_ensemble, SEEDS
)

def analyze_seed_variance():
    """Analyze variance across seeds in detail."""

    print("Loading data for analysis...")
    X_train, y_train, X_test, test_df = load_and_prepare_data()

    print("\nTraining models with all seeds for test set...")
    all_seed_probs = []
    all_seed_preds = []

    for seed_idx, seed in enumerate(SEEDS):
        print(f"  Seed {seed} ({seed_idx + 1}/{len(SEEDS)})...")
        models = train_seed_ensemble(X_train, y_train, seed)
        seed_prob = predict_seed_ensemble(models, X_test)
        seed_pred = (seed_prob >= 0.5).astype(int)

        all_seed_probs.append(seed_prob)
        all_seed_preds.append(seed_pred)

    # Convert to arrays
    all_seed_probs = np.array(all_seed_probs)  # shape: (n_seeds, n_samples)
    all_seed_preds = np.array(all_seed_preds)  # shape: (n_seeds, n_samples)

    # Calculate statistics
    final_prob = np.mean(all_seed_probs, axis=0)
    final_pred = (final_prob >= 0.5).astype(int)
    seed_variance = np.var(all_seed_probs, axis=0)
    seed_std = np.std(all_seed_probs, axis=0)

    # Agreement across seeds (how many seeds agree on the final prediction)
    agreement = np.zeros(len(final_pred))
    for i in range(len(final_pred)):
        agreement[i] = np.sum(all_seed_preds[:, i] == final_pred[i]) / len(SEEDS)

    # Create analysis dataframe
    analysis_df = pd.DataFrame({
        'PassengerId': test_df['PassengerId'].values,
        'FinalPrediction': final_pred,
        'FinalProbability': final_prob,
        'SeedVariance': seed_variance,
        'SeedStdDev': seed_std,
        'SeedAgreement': agreement,
        'MinProb': np.min(all_seed_probs, axis=0),
        'MaxProb': np.max(all_seed_probs, axis=0),
        'ProbRange': np.max(all_seed_probs, axis=0) - np.min(all_seed_probs, axis=0)
    })

    print("\n" + "="*70)
    print("SEED VARIANCE ANALYSIS")
    print("="*70)

    print(f"\nOverall Statistics:")
    print(f"  Mean probability: {final_prob.mean():.4f}")
    print(f"  Mean seed variance: {seed_variance.mean():.6f}")
    print(f"  Mean seed std dev: {seed_std.mean():.6f}")
    print(f"  Mean seed agreement: {agreement.mean():.2%}")

    print(f"\nVariance Distribution:")
    print(f"  Min variance: {seed_variance.min():.6f}")
    print(f"  25th percentile: {np.percentile(seed_variance, 25):.6f}")
    print(f"  Median variance: {np.median(seed_variance):.6f}")
    print(f"  75th percentile: {np.percentile(seed_variance, 75):.6f}")
    print(f"  Max variance: {seed_variance.max():.6f}")

    print(f"\nAgreement Distribution:")
    print(f"  Perfect agreement (100%): {np.sum(agreement == 1.0)} passengers")
    print(f"  Strong agreement (>=90%): {np.sum(agreement >= 0.9)} passengers")
    print(f"  Weak agreement (60-80%): {np.sum((agreement >= 0.6) & (agreement < 0.8))} passengers")
    print(f"  Low agreement (<60%): {np.sum(agreement < 0.6)} passengers")

    # High variance cases
    print(f"\nHigh Variance Cases (top 20):")
    high_var = analysis_df.nlargest(20, 'SeedVariance')
    print(high_var[['PassengerId', 'FinalPrediction', 'FinalProbability',
                     'SeedVariance', 'SeedAgreement', 'MinProb', 'MaxProb']].to_string(index=False))

    # Low agreement cases
    print(f"\nLow Agreement Cases (bottom 20):")
    low_agree = analysis_df.nsmallest(20, 'SeedAgreement')
    print(low_agree[['PassengerId', 'FinalPrediction', 'FinalProbability',
                      'SeedVariance', 'SeedAgreement', 'MinProb', 'MaxProb']].to_string(index=False))

    # Prediction distribution by seed
    print(f"\n" + "="*70)
    print("PREDICTION DISTRIBUTION BY SEED")
    print("="*70)

    for seed_idx, seed in enumerate(SEEDS):
        n_survived = all_seed_preds[seed_idx].sum()
        survival_rate = n_survived / len(all_seed_preds[seed_idx])
        print(f"Seed {seed:5d}: {n_survived:3d} survived ({survival_rate:.2%})")

    print(f"\nFinal Ensemble: {final_pred.sum():3d} survived ({final_pred.mean():.2%})")

    # Save detailed analysis
    analysis_path = '/home/user/kaggle-titanic-competition/approach_c_variance_analysis.csv'
    analysis_df.to_csv(analysis_path, index=False)
    print(f"\nDetailed analysis saved to: {analysis_path}")

    return analysis_df

if __name__ == "__main__":
    analysis_df = analyze_seed_variance()
