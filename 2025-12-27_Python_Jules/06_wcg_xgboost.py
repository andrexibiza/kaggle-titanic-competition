import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import re
import warnings

warnings.filterwarnings('ignore')

def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # Combine for processing
    train['is_train'] = 1
    test['is_train'] = 0
    test['Survived'] = np.nan
    full = pd.concat([train, test], sort=False).reset_index(drop=True)
    return full

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

def feature_engineering(df):
    # 1. Title
    df['Title'] = df['Name'].apply(get_title)
    # Normalize Titles
    df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don',
                                       'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    # 2. Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 3. Deck
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'M') # M for Missing

    # 4. Surname (for WCG)
    df['Surname'] = df['Name'].apply(lambda x: x.split(',')[0])

    return df

def prepare_xgb_data(df):
    # Select features for XGBoost
    # We need to encode categorical variables
    df_enc = df.copy()

    # Label Encoding
    for col in ['Sex', 'Embarked', 'Title', 'Deck', 'Surname']:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))

    # Drop non-numeric or unneeded columns for model
    drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId', 'is_train', 'Survived']
    X = df_enc.drop(drop_cols, axis=1)

    # Fill NA
    X = X.fillna(-999)

    return X, df_enc['Survived']

def get_wcg_predictions(df, xgb_preds):
    """
    Applies Woman-Child-Group logic to override XGBoost predictions.
    """
    # Define Woman-Child-Group (WCG) candidates: Females and Masters (Boys)
    # Note: 'Master' is the title for boys.

    # We need the original dataframe with 'Surname', 'Ticket', 'Title', 'Sex'
    df['Prediction'] = xgb_preds

    # Identify Woman and Child (Boys)
    df['IsWomanOrBoy'] = ((df['Title'] == 'Master') | (df['Sex'] == 'female'))

    # --- WCG Logic based on Surname ---
    # We look at the fate of the group in the TRAINING set.

    # Create a lookup for group fate in Train
    train_df = df[df['is_train'] == 1]
    test_df = df[df['is_train'] == 0]

    # 1. Surname Logic
    # Calculate survival rate of Women/Boys in each Surname group in Train
    surname_stats = train_df[train_df['IsWomanOrBoy']].groupby('Surname')['Survived'].agg(['count', 'mean', 'sum'])

    # Identify "Dead Surnames": Groups where all W/B died
    dead_surnames = surname_stats[(surname_stats['mean'] == 0.0) & (surname_stats['count'] > 0)].index.tolist()

    # Identify "Living Surnames": Groups where all W/B lived
    living_surnames = surname_stats[(surname_stats['mean'] == 1.0) & (surname_stats['count'] > 0)].index.tolist()

    # 2. Ticket Logic (usually stronger than Surname)
    ticket_stats = train_df[train_df['IsWomanOrBoy']].groupby('Ticket')['Survived'].agg(['count', 'mean', 'sum'])

    dead_tickets = ticket_stats[(ticket_stats['mean'] == 0.0) & (ticket_stats['count'] > 0)].index.tolist()
    living_tickets = ticket_stats[(ticket_stats['mean'] == 1.0) & (ticket_stats['count'] > 0)].index.tolist()

    print(f"Found {len(dead_surnames)} dead surnames and {len(living_surnames)} living surnames.")
    print(f"Found {len(dead_tickets)} dead tickets and {len(living_tickets)} living tickets.")

    # --- Apply Overrides ---

    # Iterate over Test set
    # Priority: Ticket > Surname

    final_preds = df.loc[df['is_train'] == 0, 'Prediction'].copy()

    # Helper for logging
    changes = 0

    for idx in test_df.index:
        row = df.loc[idx]
        if not row['IsWomanOrBoy']:
            continue # Skip adult males

        original_pred = row['Prediction']
        new_pred = original_pred

        # Check Ticket
        if row['Ticket'] in dead_tickets:
            new_pred = 0
        elif row['Ticket'] in living_tickets:
            new_pred = 1
        else:
            # Check Surname if Ticket didn't decide
            if row['Surname'] in dead_surnames:
                new_pred = 0
            elif row['Surname'] in living_surnames:
                new_pred = 1

        if new_pred != original_pred:
            # print(f"Overriding prediction for {row['Name']} ({row['Sex']}, {row['Title']}) from {original_pred} to {new_pred}")
            final_preds.loc[idx] = new_pred
            changes += 1

    print(f"WCG Logic modified {changes} predictions.")

    return final_preds

def main():
    # 1. Load and Prepare
    full_df = load_data()
    full_df = feature_engineering(full_df)

    X_full, _ = prepare_xgb_data(full_df)

    # Split back to train/test
    train_mask = full_df['is_train'] == 1
    X_train = X_full[train_mask]
    y_train = full_df.loc[train_mask, 'Survived']
    X_test = X_full[~train_mask]

    # 2. Train XGBoost
    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predict
    xgb_preds = model.predict(X_test)

    # 3. Apply WCG
    print("Applying WCG Post-Processing...")
    # Pass the FULL dataframe (with metadata) and the raw predictions
    # We need to map predictions back to the full_df structure or pass them aligned

    # Create a copy of full_df to store predictions for the test part
    full_df_w_preds = full_df.copy()
    full_df_w_preds.loc[~train_mask, 'Prediction'] = xgb_preds

    final_preds = get_wcg_predictions(full_df_w_preds, full_df_w_preds.loc[~train_mask, 'Prediction'])

    # 4. Create Submission
    submission = pd.DataFrame({
        'PassengerId': full_df.loc[~train_mask, 'PassengerId'],
        'Survived': final_preds.astype(int)
    })

    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    main()
