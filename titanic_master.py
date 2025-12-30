import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load data
def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

# Feature Engineering
def feature_engineering(train, test):
    # Combine dataset for consistent engineering
    data = pd.concat([train, test], sort=False)
    
    # Extract Title
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    
    # Mapping titles might be useful for models, but for WCG strings are fine or we can map later
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data['Title_Code'] = data['Title'].map(title_mapping)
    data['Title_Code'] = data['Title_Code'].fillna(0)
    
    # Extract Surname
    data['Surname'] = data['Name'].apply(lambda x: x.split(',')[0].strip())
    
    # Family Size
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    # Family Grouping (Surname + FamilySize) is a weak proxy, better to use Ticket logic for WCG
    
    return data

def run():
    print("Loading data...")
    train, test = load_data()
    
    print("Performing Feature Engineering...")
    data = feature_engineering(train, test)
    
    print("Data processed. Columns:", data.columns)

    # Split back
    train_df = data[:len(train)]
    test_df = data[len(train):]
    
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    # --- WCG LOGIC (REFINED) ---
    print("Applying Refined WCG Logic (Ticket Priority)...")
    
    # We prioritize Ticket grouping as it is more accurate than Surname.
    # WCG Rule: If a group (Family/Ticket) lives, they all live. If they die, they all die.
    
    # 1. Create Group ID based on Ticket
    # Tickets are the strongest link.
    data['Group_Ticket'] = data['Ticket'].astype(str)
    
    # Calculate survival metrics for Ticket groups
    ticket_insights = data.groupby('Group_Ticket')['Survived'].agg(['mean', 'max', 'min', 'count'])
    
    # 2. Identify 'Woman-Child' candidates (Title based is safest)
    # Master(4), Miss(2), Mrs(3) -> Likely to share fate
    wc_mask = data['Title_Code'].isin([2, 3, 4]) 
    
    # Initialize WCG Predictions
    test_df = data[len(train):].copy()
    test_df['WCG_Pred'] = np.nan
    
    # Iterate through Test set
    for idx, row in test_df.iterrows():
        # Policy 1: Check Ticket Connection
        ticket = row['Group_Ticket']
        
        if ticket in ticket_insights.index:
            # Check training data stats for this ticket
            stats = ticket_insights.loc[ticket]
            
            # If we have info on this ticket from Train set (count > 0 in Train implicitly handled by mean being non-nan)
            if not pd.isna(stats['mean']):
                # If everyone in Train with this ticket Survived -> Predict 1
                if stats['mean'] == 1.0:
                    test_df.at[idx, 'WCG_Pred'] = 1
                # If everyone in Train with this ticket Died -> Predict 0
                elif stats['mean'] == 0.0:
                    test_df.at[idx, 'WCG_Pred'] = 0
                
        # Policy 2: Fallback to Surname for W/C if Ticket didn't give a clear 0 or 1 signal?
        # Chris Deotte's strategy is mostly Ticket + Surname adjustments.
        # For now, let's stick to strict Ticket consistency to avoid false positives.
        # If Ticket is mixed (mean 0.5), we leave as NaN and let ML handle it.

    

    
    print(f"WCG predictions calculated. {test_df['WCG_Pred'].notnull().sum()} passengers determined by WCG.")

    # --- MACHINE LEARNING PIPELINE ---
    print("Training Ensemble Models for remaining passengers...")
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder

    # Preprocessing for ML
    # We need to turn string columns into numbers
    
    # 1. Sex
    data['Sex_Code'] = data['Sex'].map({'female': 1, 'male': 0}).astype(int)
    
    # 2. Embarked
    data['Embarked'] = data['Embarked'].fillna('S')
    data['Embarked_Code'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    
    # 3. Age (Fill NaNs)
    # Simple median for now, or based on Title
    data['Age'] = data['Age'].fillna(data.groupby('Title')['Age'].transform('median'))
    
    # 4. Fare (Fill NaNs)
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

    # Select features
    features = ['Pclass', 'Sex_Code', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Code', 'Title_Code', 'FamilySize']
    
    # Prepare Train/Test for ML
    # We only train on rows where we have ground truth.
    # Actually, we train on ALL training data to get the best model.
    
    X_train = data[:len(train)][features]
    y_train = data[:len(train)]['Survived']
    X_test = data[len(train):][features]
    
    # Define Models
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    
    # Voting Classifier
    # Soft voting is usually better for probabilities
    voting_clf = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('gb', gb)],
        voting='soft'
    )
    
    print("Fitting Voting Classifier...")
    voting_clf.fit(X_train, y_train)
    
    # Predict on Test
    print("Predicting...")
    ml_predictions = voting_clf.predict(X_test)
    
    # Use ML predictions where WCG is NaN
    final_predictions = test_df['WCG_Pred'].copy()
    
    # Get indices where WCG is NaN
    mask = final_predictions.isnull()
    
    # Fill with ML predictions
    # Note: final_predictions is a Series with the same index as X_test
    # We need to be careful with alignment.
    # X_test corresponds exactly to the test_df rows.
    
    final_predictions[mask] = ml_predictions[mask]
    
    # Ensure no NaNs remain
    if final_predictions.isnull().sum() > 0:
        print("WARNING: NaNs in prediction. Filling with 0 (Dead).")
        final_predictions = final_predictions.fillna(0)
    
    # Create final submission
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": final_predictions.astype(int)
    })
    
    # STRICT VERIFICATION
    expected_rows = 418
    if len(submission) != expected_rows:
        raise ValueError(f"Submission has {len(submission)} rows, expected {expected_rows}!")
    
    print(f"Submission generated with {len(submission)} rows. Verified.")
    submission.to_csv('submission.csv', index=False)
    
    # Post-processing: Remove trailing newline to satisfy strict "no blank row" requirement
    with open('submission.csv', 'rb') as f:
        content = f.read().strip()
    with open('submission.csv', 'wb') as f:
        f.write(content)
        
    print("Trailing whitespace removed.")

if __name__ == "__main__":
    run()
