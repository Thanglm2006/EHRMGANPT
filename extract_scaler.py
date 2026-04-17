import pandas as pd
import numpy as np
import pickle
import os

# Paths from your environment
DATA_FILEPATH = r'D:\vecna\mimic_extract\output\all_hourly_data.h5'
OUTPUT_PATH = r'D:\vecna\EHR-M-GAN\Data\Mimic3\clinical_scaler.pkl'

def extract_normalization_parameters():
    print(f"Loading raw vital sign data from: {DATA_FILEPATH}")
    print("This may take a minute due to file size (7.6GB)...")
    
    # 1. Load the raw vitals table
    vital = pd.read_hdf(DATA_FILEPATH, 'vitals_labs')
    
    # 2. Match the column selection logic from the preprocessing script
    idx = pd.IndexSlice
    vital = vital.loc[:, idx[:, 'mean']]
    vital = vital.droplevel(1, axis=1)
    
    # 3. Handle identical column dropping logic
    # In preprocessing, columns with only NaNs are dropped.
    # We do this to ensure we have exactly 104 features.
    vital_clean = vital.dropna(axis=1, how='all')
    
    n_features = len(vital_clean.columns)
    print(f"Identified {n_features} clinical features.")
    
    if n_features != 104:
        print(f"WARNING: Feature count ({n_features}) does not match expected 104.")
        print("Continuing anyway, but you should verify feature alignment.")

    # 4. Calculate clinical min and max
    # These are the values needed to un-normalize the GAN output
    mins = vital_clean.min().values
    maxes = vital_clean.max().values
    feature_names = vital_clean.columns.tolist()

    # 5. Capture Discrete Feature names (Interventions)
    discrete_names = [
        'vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 
        'isuprel', 'milrinone', 'norepinephrine', 'phenylephrine', 
        'vasopressin', 'colloid_bolus', 'crystalloid_bolus'
    ]

    # 6. Save the parameters
    scaler_data = {
        'mins': mins,
        'maxes': maxes,
        'feature_names': feature_names,
        'discrete_names': discrete_names
    }

    print(f"Saving clinical scaler to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(scaler_data, f)
    
    print("Success! You can now use this file in test.py to see real clinical units.")

if __name__ == '__main__':
    extract_normalization_parameters()
