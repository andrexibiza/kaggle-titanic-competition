#!/usr/bin/env python
# coding: utf-8

# # Titanic - Phase 1: Model Benchmarking
#
# This notebook benchmarks multiple state-of-the-art models on the replicated 'V4' feature set.
# Goal: Identify the highest performing single model using rigorous Cross-Validation.

# In[ ]:


import pandas as pd
import numpy as np
import titanic_utils
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


# ## 1. Load and Prepare Data

# In[ ]:


X_train, y_train, X_test, test_ids = titanic_utils.prepare_v4_data()

print(f"Training Shape: {X_train.shape}")
print(f"Test Shape: {X_test.shape}")
X_train.head()


# ## 2. Define Models

# In[ ]:


models = {}

# Random Forest (Baseline)
models['RF'] = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# XGBoost
models['XGB'] = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

# LightGBM
models['LGBM'] = LGBMClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)

# CatBoost
# CatBoost (Skipped due to sklearn compatibility issues in this environment)
# models['CatBoost'] = CatBoostClassifier(
#    iterations=100,
#    depth=3,
#    learning_rate=0.1,
#    verbose=0,
#    random_state=42
# )

# SVM (Needs Scaling)
models['SVM'] = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42))
])

# Logistic Regression (Needs Scaling)
models['LR'] = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(random_state=42))
])


# ## 3. Run Benchmarking

# In[ ]:


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = {}
print("Starting Cross-Validation...\n")

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    results[name] = scores
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")


# ## 4. Train Best Model & Generate Submission

# In[ ]:


# Find best model
best_name = max(results, key=lambda k: results[k].mean())
print(f"\nBest Model: {best_name}")

# Retrain on full data
best_model = models[best_name]
best_model.fit(X_train, y_train)

# Predict
predictions = best_model.predict(X_test)

# Save
submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': predictions})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved.")
