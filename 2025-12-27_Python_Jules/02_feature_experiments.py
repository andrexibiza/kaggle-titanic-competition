#!/usr/bin/env python
# coding: utf-8

# # Titanic - Phase 2: Feature Engineering Experiments
#
# Baseline: LightGBM (CV: 0.8428)
# Experiments:
# 1. **FarePerPerson**: Adjusting Fare by Ticket Frequency.
# 2. **TicketGroupSize**: Explicit group size feature.
# 3. **WCG Feature**: Encoding the Woman-Child-Group logic as a feature.

# In[ ]:


import pandas as pd
import numpy as np
import titanic_utils
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')


# ## 1. Load Baseline Data

# In[ ]:


# We need to access the raw data first to engineer new features before getting the clean matrix
# But titanic_utils.prepare_v4_data does everything at once.
# Let's modify/extend titanic_utils or just do it here manually on top of the utils.


# In[ ]:


train, test = titanic_utils.load_data()
full = pd.concat([train.drop('Survived', axis=1), test], axis=0, ignore_index=True)

# Calculate Ticket Frequency
ticket_counts = full['Ticket'].value_counts().to_dict()
full['TicketGroupSize'] = full['Ticket'].map(ticket_counts)

# Calculate Fare Per Person
# First fill NA Fare
full['Fare'] = full['Fare'].fillna(full['Fare'].median())
full['FarePerPerson'] = full['Fare'] / full['TicketGroupSize']

# Now we need to merge these into the processed data
# Simplest way: Call prepare_v4_data, then add these columns back by index.
X_train, y_train, X_test, test_ids = titanic_utils.prepare_v4_data()

# Re-align indices
full_train_features = full.iloc[:len(X_train)][['TicketGroupSize', 'FarePerPerson']].reset_index(drop=True)
full_test_features = full.iloc[len(X_train):][['TicketGroupSize', 'FarePerPerson']].reset_index(drop=True)

# Add to X_train, X_test
X_train_v2 = pd.concat([X_train.reset_index(drop=True), full_train_features], axis=1)
X_test_v2 = pd.concat([X_test.reset_index(drop=True), full_test_features], axis=1)

print("New Train Shape:", X_train_v2.shape)


# ## 2. Test Improvements

# In[ ]:


model = LGBMClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("Baseline (V4 features only):")
scores_base = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"{scores_base.mean():.4f} (+/- {scores_base.std():.4f})")

print("\nWith FarePerPerson + TicketGroupSize:")
scores_v2 = cross_val_score(model, X_train_v2, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"{scores_v2.mean():.4f} (+/- {scores_v2.std():.4f})")


# ## 3. Experiment: WCG Feature
# Encode the WCG logic as a feature: -1 (Die), 1 (Survive), 0 (Unknown).

# In[ ]:


# Replicate logic from titanic_wcg_model.R
# GroupId = Surname + Pclass + Ticket
# But wait, we already have FamilySurvived and TicketSurvived in V4 features.
# V4 features: 'FamilySurvived', 'TicketSurvived', 'GroupSurvived'
# These are continuous means (0.0 to 1.0).
# The WCG logic is essentially thresholding these to 0 or 1 for Females/Boys.
# LightGBM should theoretically learn this interaction (Sex=Female & GroupSurvived=0 -> Die).
# However, let's try explicitly making it a categorical feature.

# Check correlations
corr_matrix = X_train_v2.corrwith(y_train).sort_values(ascending=False)
print("\nCorrelations with Target:")
print(corr_matrix.head(10))


# In[ ]:


# If improvement found, update submission
if scores_v2.mean() > scores_base.mean():
    print("\nImprovement found! Retraining...")
    model.fit(X_train_v2, y_train)
    preds = model.predict(X_test_v2)
    pd.DataFrame({'PassengerId': test_ids, 'Survived': preds}).to_csv("submission_v2_features.csv", index=False)
    print("Saved submission_v2_features.csv")
else:
    print("\nNo improvement.")
