import os
import numpy as np
import pandas as pd
from scipy.signal import resample

from engineered_features import create_features

def load_device_signals(participant_id, data_root="pilot", device="empatica", physioparam="bvp", condition="baseline"):
    """
    Loads BVP signals for a single participant and condition.
    
    Args:
        participant_id (str or int): Participant folder name
        data_root (str): Top-level folder (e.g., 'pilot')
        condition (str): 'baseline' or 'cognitive_load'
        
    Returns:
        dict: {'bvp': ..., 'eda': ..., 'temp': ...}, each a 1D np.array aligned at 4 Hz
    """
    
    devices = ["empatica", "samsung", "muse"]

    pid = str(participant_id)
    base_path = os.path.join(data_root, pid, condition)

    
    ebvp = pd.read_csv(os.path.join(base_path, "empatica_bvp.csv"))["bvp"].values
    # Trim 2 sec at start & end from BVP to match EDA/TEMP preprocessing
    sec_remove = 2
    ebvp = ebvp[64 * sec_remove : -64 * sec_remove]  # 64 Hz

    return {
        "bvp": ebvp
    }

def create_windows(signal, window_size=256, step_size=64):
    windows = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        window = signal[start:start+window_size]
        windows.append(window)
    return np.stack(windows)

def create_windows_and_features(participant_id, condition, window_size, step_size):
    s_t = load_device_signals(participant_id=participant_id, condition=condition)["bvp"]
    s_t_windows = create_windows(s_t, window_size=window_size, step_size=step_size)
    
    features = np.array([list(create_features(window).values()) for window in s_t_windows])
    feature_names = list(create_features(s_t_windows[0]).keys())
    
    return features, feature_names

def create_full_dataset_for_participant(participant_id, window_size, step_size):
    features_b, feature_names = create_windows_and_features(participant_id, 'baseline', window_size, step_size)
    features_c, _ = create_windows_and_features(participant_id, 'cognitive_load', window_size, step_size)
    
    labels_b = np.zeros(len(features_b))  # Baseline = 0
    labels_c = np.ones(len(features_c))   # Cognitive Load = 1
    
    # Combine
    X = np.vstack([features_b, features_c])
    y = np.concatenate([labels_b, labels_c])
    
    return X, y, feature_names