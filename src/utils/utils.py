"""
Utility functions
"""
import pickle
import pandas as pd

def load_pickle(filepath):
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj, filepath):
    """Save object to pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
