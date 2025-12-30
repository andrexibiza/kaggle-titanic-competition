import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the directory to path to import the module
sys.path.append(os.path.abspath('2025-12-27_Python_Jules'))

from importlib.util import spec_from_file_location, module_from_spec

# Load the module dynamically since it starts with a number
spec = spec_from_file_location("wcg_module", "2025-12-27_Python_Jules/06_wcg_xgboost.py")
wcg_module = module_from_spec(spec)
spec.loader.exec_module(wcg_module)

class TestWCGLogic(unittest.TestCase):
    def test_wcg_logic(self):
        # Create a dummy dataframe
        # Columns: Surname, Ticket, Title, Sex, is_train, Survived

        data = {
            'PassengerId': [1, 2, 3, 4, 5, 6],
            'Surname': ['Smith', 'Smith', 'Smith', 'Doe', 'Doe', 'Jones'],
            'Ticket':  ['T1',    'T1',    'T1',    'T2',  'T2',  'T3'],
            'Title':   ['Mrs',   'Master','Mr',    'Mrs', 'Master', 'Miss'],
            'Sex':     ['female','male',  'male',  'female','male', 'female'],
            'is_train':[1,       1,       1,       1,     0,      0],
            'Survived':[1,       1,       0,       0,     np.nan, np.nan],
            'Prediction': [1,    1,       0,       0,     1,      1] # Mock predictions
        }

        df = pd.DataFrame(data)

        # Test Case 1: Smith Family (Train)
        # Mrs. Smith (Surv=1) and Master Smith (Surv=1).
        # Mr. Smith (Surv=0) - adult male, ignored by WCG logic usually unless specifically checking everyone.
        # WCG check: Women/Boys in Smith/T1 in Train: Mrs (1), Master (1). Mean = 1.0. All Lived.

        # Test Case 2: Doe Family
        # Mrs. Doe (Surv=0) in Train.
        # Master Doe in Test.
        # WCG check: Women/Boys in Doe/T2 in Train: Mrs (0). Mean = 0.0. All Died.
        # Master Doe prediction is 1. Should be overridden to 0.

        # Test Case 3: Jones
        # Miss Jones in Test. No Train info. Prediction 1. Should stay 1.

        # Run WCG
        final_preds = wcg_module.get_wcg_predictions(df, df.loc[df['is_train']==0, 'Prediction'])

        # Check Master Doe (ID 5)
        # Prediction was 1. Should be 0 because Mrs Doe died.
        self.assertEqual(final_preds.loc[4], 0, "Master Doe should be corrected to 0 (Dead) because his family died.")

        # Check Miss Jones (ID 5) -> Index 5
        self.assertEqual(final_preds.loc[5], 1, "Miss Jones should remain 1 (Survive) because no info.")

if __name__ == '__main__':
    unittest.main()
