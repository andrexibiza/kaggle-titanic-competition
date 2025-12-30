import os
import sys

# Set the token provided by the user
# Inferring credentials:
# Username from path: andrexibiza
# Key from user provided JSON
os.environ['KAGGLE_USERNAME'] = "andrexibiza"
os.environ['KAGGLE_KEY'] = "eba93f5470e1e4c38f26ad0419f8ce28"

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    print("Kaggle API imported.")
    
    api = KaggleApi()
    print("Authenticating...")
    try:
        api.authenticate()
        print("Authenticated successfully.")
    except Exception as e:
        print(f"Authentication failed: {e}")
        # Continue to see if it works anyway if the token is handled deeply
        
    print("Submitting...")
    api.competition_submit('submission.csv', 'WCG + Ensemble Voting', 'titanic')
    print("Submission successful!")
    
except Exception as e:
    print(f"An error occurred: {e}")
