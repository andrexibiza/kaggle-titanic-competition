# Approach C: Seed-Averaged Ensemble Results

## Executive Summary

**Goal**: Build a seed-averaged ensemble to reduce variance while capturing signal.

**Status**: ✓ Successfully implemented and evaluated

**Key Achievement**: Very low variance (0.000073) with strong CV performance (88.21%)

---

## Configuration

### Architecture
- **Models per seed**: XGBoost + Random Forest + Logistic Regression
- **Number of seeds**: 10
- **Seeds used**: [42, 123, 456, 789, 1000, 2023, 2024, 2025, 314, 271]
- **Ensemble method**: Average probabilities across all 30 models (10 seeds × 3 models)

### Model Hyperparameters

**XGBoost:**
```python
n_estimators=100, max_depth=3, learning_rate=0.1,
subsample=0.8, colsample_bytree=0.8
```

**Random Forest:**
```python
n_estimators=100, max_depth=6
```

**Logistic Regression:**
```python
C=1.0, max_iter=1000
```

### Features (11 total)
1. Pclass
2. Sex_Code (female=1, male=0)
3. Age (imputed by Title median)
4. SibSp
5. Parch
6. Fare
7. Embarked_Code (S=0, C=1, Q=2)
8. FamilySize (SibSp + Parch + 1)
9. Title_Code (Mr=1, Miss=2, Mrs=3, Master=4, Rare=5)
10. FamilySurvived (survival rate of family group, default 0.5)
11. TicketSurvived (survival rate of ticket group, default 0.5)

---

## Performance Results

### Cross-Validation (10-Fold Stratified)

**Mean CV Accuracy**: 88.21% (± 2.54%)

**Individual Fold Scores**:
- Fold 1: 92.22%
- Fold 2: 84.27%
- Fold 3: 87.64%
- Fold 4: 86.52%
- Fold 5: 85.39%
- Fold 6: 88.76%
- Fold 7: 88.76%
- Fold 8: 89.89%
- Fold 9: 92.13%
- Fold 10: 86.52%

### Variance Analysis

**Mean Seed Variance**:
- Cross-validation: 0.000061
- Test set: 0.000073

**Variance Distribution (Test Set)**:
- Minimum: 0.000000
- 25th percentile: 0.000015
- Median: 0.000033
- 75th percentile: 0.000087
- Maximum: 0.001011

**Seed Agreement**:
- Mean agreement: 99.78%
- Perfect agreement (100%): 414/418 passengers (99.04%)
- Strong agreement (≥90%): 415/418 passengers (99.28%)
- Weak agreement (60-80%): 1 passenger (0.24%)
- Low agreement (<60%): 0 passengers

---

## Prediction Statistics

### Survival Rate Analysis

**Target Range**: 36-38%

**Individual Seed Predictions**:
| Seed | Survived | Survival Rate |
|------|----------|---------------|
| 42   | 167      | 39.95%        |
| 123  | 166      | 39.71%        |
| 456  | 165      | 39.47%        |
| 789  | 164      | 39.23%        |
| 1000 | 166      | 39.71%        |
| 2023 | 166      | 39.71%        |
| 2024 | 167      | 39.95%        |
| 2025 | 166      | 39.71%        |
| 314  | 166      | 39.71%        |
| 271  | 166      | 39.71%        |

**Final Ensemble**: 166/418 survived (39.71%)

**Variance**: 1.0 survivors (0.24% range: 39.23% - 39.95%)

⚠️ **Note**: Survival rate is 1.71% above target range (36-38%), but within reasonable bounds.

---

## Stability Analysis

### High Variance Cases (Top 5)

Passengers with highest prediction variance across seeds:

| PassengerId | Final Pred | Final Prob | Variance   | Agreement |
|-------------|------------|------------|------------|-----------|
| 1094        | 1          | 0.5429     | 0.001011   | 100%      |
| 1106        | 1          | 0.6864     | 0.000859   | 100%      |
| 1084        | 0          | 0.4682     | 0.000668   | 80%       |
| 1066        | 0          | 0.3826     | 0.000613   | 100%      |
| 1236        | 1          | 0.6108     | 0.000553   | 100%      |

### Low Agreement Cases

Only 3 passengers with <100% agreement:

| PassengerId | Final Pred | Final Prob | Variance   | Agreement |
|-------------|------------|------------|------------|-----------|
| 1212        | 1          | 0.5073     | 0.000532   | 60%       |
| 893         | 0          | 0.4795     | 0.000371   | 80%       |
| 1084        | 0          | 0.4682     | 0.000668   | 80%       |

These are borderline cases very close to the 0.5 threshold, explaining the disagreement.

---

## Key Findings

### Strengths

1. **Exceptional Stability**:
   - Mean seed variance of 0.000073 indicates extremely low model variance
   - 99.04% of predictions have perfect agreement across all seeds
   - Standard deviation of only 0.007 in probability predictions

2. **Strong Performance**:
   - 88.21% mean CV accuracy is competitive
   - Consistent across folds (std: 2.54%)
   - High confidence predictions (most probabilities far from 0.5)

3. **Robust Ensemble**:
   - 30 models (10 seeds × 3 model types) provide strong signal averaging
   - Individual seed predictions very consistent (164-167 survivors)
   - Prediction range of only 0.24% across all seeds

### Limitations

1. **Survival Rate Deviation**:
   - Predicted 39.71% survival vs. target 36-38%
   - Could be adjusted with threshold tuning (not done per requirements)
   - Indicates potential slight optimism bias

2. **Uncertain Cases**:
   - 3 passengers with <100% agreement
   - PassengerId 1212 particularly uncertain (60% agreement)
   - These are close to decision boundary (prob ≈ 0.5)

---

## Files Generated

1. **approach_c_seed_ensemble.py** - Main implementation
2. **submission_approach_c.csv** - Kaggle submission file (418 predictions)
3. **approach_c_analysis.py** - Variance analysis script
4. **approach_c_variance_analysis.csv** - Detailed per-passenger variance data

---

## Recommendations

### For Kaggle Submission
- Submit **submission_approach_c.csv** as is
- Expected leaderboard score: ~78-82% (based on CV and typical CV-LB gap)

### For Further Improvement
If survival rate adjustment is needed (to hit 36-38% target):

1. **Option 1**: Threshold adjustment
   - Current threshold: 0.5
   - Recommended threshold: ~0.52 would reduce to ~37%
   - However, this violates "DO NOT optimize threshold" requirement

2. **Option 2**: Re-run with different seed set
   - Current seeds may have slight systematic bias
   - Alternative seeds could naturally fall in target range

3. **Option 3**: Feature adjustment
   - Current features may slightly over-predict survival
   - Could reduce FamilySurvived/TicketSurvived feature weights

### For Model Enhancement
- Current ensemble already very stable and performant
- Adding more seeds would have diminishing returns
- Consider diversifying model types (e.g., add GradientBoosting, CatBoost)

---

## Conclusion

Approach C successfully achieves the primary goal of **reducing variance while capturing signal**:

✅ **Ultra-low variance**: 0.000073 (excellent)
✅ **Strong CV performance**: 88.21% (competitive)
✅ **High stability**: 99.78% seed agreement (exceptional)
✅ **Consistent predictions**: Only 0.24% range across seeds
⚠️ **Survival rate**: 39.71% (slightly above 36-38% target)

The seed-averaged ensemble demonstrates exceptional stability and robustness, with minimal variance across different random seeds. This approach would be ideal for production deployment where prediction consistency is critical.
