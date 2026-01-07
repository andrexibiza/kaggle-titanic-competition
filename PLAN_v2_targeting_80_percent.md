# Plan: Targeting 80-85% Accuracy on Kaggle Titanic

## Executive Summary

**Current Best**: 0.78947 (V4 - Simple 3-model ensemble)
**Target**: 0.80+ (break through 80% barrier)
**Key Insight**: Simplicity wins on small data. The advanced solution (0.74401) failed due to over-engineering.

---

## Part 1: Root Cause Analysis

### Why V4 Succeeded (0.78947)
1. **Simple architecture**: XGBoost + Random Forest + GLMnet (3 diverse models)
2. **Conservative hyperparameters**: max_depth=3 (prevents overfitting)
3. **Minimal features**: ~10-14 features (not 39!)
4. **Simple averaging**: Equal weights, no learned blending
5. **No overrides**: Trusts the model, no WCG/rule-based hacks
6. **Standard threshold**: 0.5 (not optimized)

### Why Advanced Solution Failed (0.74401)
1. **Feature explosion**: 39 features on 891 samples = overfitting
2. **Model redundancy**: 8 models with correlated errors
3. **Threshold overfitting**: 0.32 threshold (huge red flag!)
4. **Over-optimistic**: Predicted 45% survival vs actual ~37%
5. **WCG blending**: Added noise, not signal
6. **No ablation**: No validation that each component helped

### The Small Data Paradox
- 891 training samples, 418 test samples
- Flipping 4 passengers = ~1% score change
- **Variance is the enemy, not bias**
- Complex models memorize noise

---

## Part 2: Proposed Approaches (In Order of Priority)

### Approach A: Refined V4 (Conservative - Target: 0.79-0.80)

**Philosophy**: Take what works, make minimal targeted improvements.

**Changes from V4**:
1. Keep exact same 3-model architecture (XGB + RF + GLM)
2. Keep exact same hyperparameters (max_depth=3)
3. Add ONE carefully validated feature at a time
4. Test each addition with repeated CV (not single split)
5. Only keep features that improve CV by >0.5%

**Candidate Features to Test (one at a time)**:
- `FarePerPerson` = Fare / TicketGroupSize
- `IsAlone` = (FamilySize == 1)
- `Title_Sex` interaction (encode Mrs/Miss differently)

**Validation Protocol**:
- 10-fold CV, repeated 5 times (50 total folds)
- Only accept changes with p < 0.05 improvement

---

### Approach B: SVM + Calibration (Alternative - Target: 0.79-0.80)

**Philosophy**: Web research suggests SVM works well on small datasets.

**Architecture**:
1. **SVM with RBF kernel** (standardized features)
2. **Calibrated probabilities** via CalibratedClassifierCV
3. **Simple ensemble**: SVM + RF + Logistic Regression

**Key Settings**:
- Use ONLY core features: Sex, Pclass, Age, Fare, FamilySize, Title
- StandardScaler on all numeric features
- C and gamma tuned via nested CV (not Optuna - too aggressive)

**Why This Might Work**:
- SVM has strong regularization built-in
- RBF kernel captures non-linear patterns without deep trees
- Less prone to overfitting than gradient boosting

---

### Approach C: Seed Ensemble with Feature Selection (Robust - Target: 0.79-0.80)

**Philosophy**: V11 (seed averaging) was more honest but lost lucky variance. Can we get best of both?

**Architecture**:
1. Train V4 architecture across 10 seeds
2. For each seed, also train with feature subsets (bootstrap features)
3. Select predictions where models agree (high confidence)
4. Use V4 single-seed for disagreement cases

**Key Innovation**:
- Consensus voting on "easy" cases (models agree)
- Lucky single-seed prediction on "hard" cases (models disagree)

---

### Approach D: Error Analysis Targeting (Surgical - Target: 0.80+)

**Philosophy**: Find the ~8-10 passengers V4 gets wrong and fix just those.

**Method**:
1. Identify V4's likely errors via ensemble disagreement analysis
2. Build specialized micro-models for edge cases:
   - Class 3 females with large families
   - Class 1 males traveling alone
   - Young males (age 12-18)
3. Override V4 only on these specific subgroups

**Risk**: May overfit to leaderboard structure. Requires careful validation.

---

## Part 3: Feature Engineering Guidelines

### KEEP (Proven Valuable)
| Feature | Importance | Notes |
|---------|------------|-------|
| Sex | Critical | Women survive 74%, men 19% |
| Pclass | Critical | Class 1: 63%, Class 3: 24% |
| Title | High | Master (boys) is key signal |
| Age | Medium | Impute by Title median |
| Fare | Medium | Proxy for wealth/deck |
| FamilySize | Medium | 2-4 optimal, 1 and 5+ worse |
| FamilySurvived | High | 0.5 default acts as flag |
| TicketSurvived | High | Group survival pattern |

### REMOVE (Caused Overfitting)
| Feature | Problem |
|---------|---------|
| WCG_Score | Circular logic, encodes answer |
| 15+ interaction features | Too many for 891 samples |
| Optimized threshold | Overfits to train split |
| Multiple survival rate variants | Redundant, confuses model |

### TEST CAREFULLY (May Help)
| Feature | Hypothesis | Validation Needed |
|---------|------------|-------------------|
| FarePerPerson | Corrects for group tickets | CV improvement >0.5% |
| Deck | Cabin location matters | Only if not too sparse |
| Embarked | Port differences | Marginal, likely noise |

---

## Part 4: Model Configuration Guidelines

### XGBoost (Conservative)
```python
XGBClassifier(
    n_estimators=100,      # Not 200-500
    max_depth=3,           # CRITICAL: Keep shallow
    learning_rate=0.1,     # Not 0.01-0.05
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    random_state=42
)
```

### Random Forest (Conservative)
```python
RandomForestClassifier(
    n_estimators=100,      # Not 200-500
    max_depth=6,           # Moderate depth
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

### Logistic Regression (Regularized)
```python
LogisticRegression(
    C=1.0,                 # Moderate regularization
    penalty='l2',
    max_iter=1000,
    random_state=42
)
```

### Ensemble Combination
```python
# Simple average - DO NOT optimize weights
final_prob = (prob_xgb + prob_rf + prob_lr) / 3
final_pred = (final_prob >= 0.5).astype(int)  # Standard threshold
```

---

## Part 5: Validation Protocol

### DO
- Use RepeatedStratifiedKFold (10 folds × 5 repeats)
- Report mean ± std across all 50 evaluations
- Only accept improvements with statistical significance
- Keep survival rate prediction ~36-38%

### DON'T
- Optimize threshold on training data
- Use single CV split for decisions
- Add features without ablation study
- Trust Optuna/Bayesian optimization on small data

### Sanity Checks Before Submission
1. Predicted survival rate: Should be 36-38% (not 45%!)
2. Class 1 female survival: ~95%+
3. Class 3 male survival: ~15%
4. Master (boys) survival: ~50-60%

---

## Part 6: Implementation Roadmap

### Phase 1: Reproduce V4 Baseline (Day 1)
- [ ] Implement exact V4 in Python
- [ ] Verify CV score matches (~0.84)
- [ ] Verify submission matches 0.78947

### Phase 2: Ablation Study (Day 1-2)
- [ ] Test removing each feature one at a time
- [ ] Test adding FarePerPerson
- [ ] Test adding IsAlone
- [ ] Document impact of each change

### Phase 3: Alternative Models (Day 2)
- [ ] Test SVM with RBF kernel
- [ ] Test calibrated ensemble
- [ ] Compare CV scores

### Phase 4: Error Analysis (Day 2-3)
- [ ] Identify high-disagreement passengers
- [ ] Analyze patterns in disagreements
- [ ] Test targeted micro-models

### Phase 5: Final Submission (Day 3)
- [ ] Select best approach based on CV
- [ ] Verify survival rate ~37%
- [ ] Submit and compare to 0.78947

---

## Part 7: Success Criteria

### Minimum Success (Match V4)
- CV: 0.84+
- Leaderboard: 0.789+
- Survival rate: 36-38%

### Target Success (Beat V4)
- CV: 0.85+
- Leaderboard: 0.80+
- Statistical significance: p < 0.05

### Stretch Goal
- Leaderboard: 0.82+
- Would require novel insight or data augmentation

---

## Part 8: Risk Mitigation

### Risk 1: Overfitting to Leaderboard
**Mitigation**: Use repeated CV, don't optimize to public score

### Risk 2: Feature Engineering Creep
**Mitigation**: Maximum 15 features, ablation required

### Risk 3: Model Complexity Creep
**Mitigation**: Maximum 3-4 models in ensemble

### Risk 4: Threshold Manipulation
**Mitigation**: Always use 0.5 threshold

---

## Appendix: Key Lessons from Failed Approaches

| Approach | Score | Lesson |
|----------|-------|--------|
| WCG Override | 0.756 | Hard rules overfit to training patterns |
| Deep Learning | 0.775 | 891 samples insufficient for DNNs |
| Pseudo-labeling | 0.758 | Feedback loops amplify errors |
| Complex Stacking | 0.773 | Meta-learner can't learn on 891 samples |
| Advanced Hybrid | 0.744 | Everything wrong: 39 features, 8 models, 0.32 threshold |

---

## Sources

- [Towards Data Science: Top 7% Titanic Solution](https://towardsdatascience.com/introduction-to-kaggle-and-scoring-top-7-in-the-titanic-competition-7a29ce9c24ae/)
- [Kaggle: Titanic Solution Top 8%](https://www.kaggle.com/code/akhileshthite/titanic-solution-top-8)
- [LinkedIn: Top 10% Titanic Approach](https://www.linkedin.com/pulse/how-you-approach-titanic-problem-kaggle-places-easily-vaishal-shah)
- [Kaggle: Top 1% Titanic Solution](https://www.kaggle.com/code/nikitakudriashov/top-1-titanic-solution)
