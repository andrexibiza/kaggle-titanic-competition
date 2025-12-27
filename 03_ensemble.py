#!/usr/bin/env python
# coding: utf-8

# # Titanic - Phase 3: Ensembling
#
# Goal: Combine best models (LGBM, XGB, SVM, RF) to push accuracy beyond 0.8428.

# In[ ]:


import pandas as pd
import numpy as np
import titanic_utils
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Load Data
X_train, y_train, X_test, test_ids = titanic_utils.prepare_v4_data()


# In[ ]:


# Define Base Models

# 1. LGBM
clf1 = LGBMClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1
)

# 2. XGB
clf2 = XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.1,
    eval_metric='logloss', use_label_encoder=False, random_state=42
)

# 3. SVM (Scaled)
clf3 = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42))
])

# 4. RF
clf4 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

estimators = [
    ('lgbm', clf1),
    ('xgb', clf2),
    ('svm', clf3),
    ('rf', clf4)
]


# ## 1. Soft Voting Classifier

# In[ ]:


voting_clf = VotingClassifier(estimators=estimators, voting='soft')

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores_voting = cross_val_score(voting_clf, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

print(f"Voting CV: {scores_voting.mean():.4f} (+/- {scores_voting.std():.4f})")


# ## 2. Stacking Classifier (Logistic Regression Meta)

# In[ ]:


stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=10,  # Internal CV for generating OOF
    n_jobs=-1
)

# Note: Evaluating Stacking via outer CV is computationally expensive (nested CV).
# We will do a simple 5-fold outer CV to check.
scores_stack = cross_val_score(stacking_clf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

print(f"Stacking CV (5-fold): {scores_stack.mean():.4f} (+/- {scores_stack.std():.4f})")


# In[ ]:


# Decide Winner and Save
if scores_stack.mean() > scores_voting.mean():
    print("Best: Stacking")
    final_model = stacking_clf
    filename = "submission_stacking.csv"
else:
    print("Best: Voting")
    final_model = voting_clf
    filename = "submission_voting.csv"

final_model.fit(X_train, y_train)
preds = final_model.predict(X_test)

pd.DataFrame({'PassengerId': test_ids, 'Survived': preds}).to_csv(filename, index=False)
print(f"Saved {filename}")
