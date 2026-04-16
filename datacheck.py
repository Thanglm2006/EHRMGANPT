import pickle
import numpy as np

with open(r'Data/Mimic3/vital_sign_24hrs.pkl', 'rb') as f:
    vitals = pickle.load(f)

print("1. Shape of Vitals:", vitals.shape)

has_nan = np.isnan(vitals).any()
print("2. Contains NaN?:", "NaN found!" if has_nan else "Clean")

print("3. min:", np.nanmin(vitals))
print("4. max:", np.nanmax(vitals))

print(vitals[0])

with open(r'Data/Mimic3/med_interv_24hrs.pkl', 'rb') as f:
    med_interv_24hrs = pickle.load(f)

print("1. Shape of med_interv_24hrs:", med_interv_24hrs.shape)

has_nan = np.isnan(med_interv_24hrs).any()
print("2. Contains NaN?:", "NaN found!" if has_nan else "Clean")

print("3. min:", np.nanmin(med_interv_24hrs))
print("4. max:", np.nanmax(med_interv_24hrs))

active_indices = np.where(np.sum(med_interv_24hrs, axis=(1, 2)) > 0)[0]

example_idx = active_indices[12]

print(f"Example patient at index: {example_idx}")
print("Intervention matrix (24 hours x number of features):")
print(med_interv_24hrs[example_idx])

print(med_interv_24hrs.shape[2])