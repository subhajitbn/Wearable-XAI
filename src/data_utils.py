import os
import numpy as np
import pandas as pd
from scipy.signal import resample

from .engineered_features import create_features
from .utils import create_windows

def load_device_signals(participant_id, data_folder = "../data/", data_root="pilot", device="empatica", physioparam="bvp", condition="baseline"):
    """
    Load and preprocess physiological signals from specified device for a given participant.

    Parameters:
    participant_id (int or str): The ID of the participant whose data to load.
    data_root (str): The root directory where participant data is stored. Default is "pilot".
    device (str): The device from which to load data. Options are "empatica", "samsung", "muse".
    physioparam (str): The physiological parameter to load. Default is "bvp".
    condition (str): The condition under which data was collected. Default is "baseline".

    Returns:
    dict: A dictionary containing preprocessed physiological data, currently only "bvp" data is returned.
    """

    devices = ["empatica", "samsung", "muse"]

    pid = str(participant_id)
    base_path = os.path.join(data_folder, data_root, pid, condition)

    
    ebvp = pd.read_csv(os.path.join(base_path, "empatica_bvp.csv"))["bvp"].values
    # Trim 5 sec at start & end from BVP to match EDA/TEMP preprocessing
    sec_remove = 5
    ebvp = ebvp[64 * sec_remove : -64 * sec_remove]  # 64 Hz

    return {
        "bvp": ebvp
    }


def create_windows_and_features(participant_id, condition, window_size, step_size):
    """
    Generate features and feature names from physiological signal windows for a given participant and condition.

    Parameters:
    participant_id (int or str): The ID of the participant whose data to process.
    condition (str): The condition under which data was collected, e.g., 'baseline' or 'cognitive_load'.
    window_size (int): The size of each window for signal segmentation.
    step_size (int): The step size between each window.

    Returns:
    tuple: A tuple containing:
        - features (2D array): An array of extracted features for each window.
        - feature_names (list): A list of feature names extracted from each window.
    """

    s_t = load_device_signals(participant_id=participant_id, condition=condition)["bvp"]
    s_t_windows = create_windows(s_t, window_size=window_size, step_size=step_size)
    
    features = np.array([list(create_features(window).values()) for window in s_t_windows])
    feature_names = list(create_features(s_t_windows[0]).keys())
    
    return features, feature_names

def create_full_dataset_for_participant(participant_id, window_size, step_size):
    """
    Generate a full dataset for a given participant by combining baseline and cognitive load datasets.

    Parameters:
    participant_id (int or str): The ID of the participant whose data to process.
    window_size (int): The size of each window for signal segmentation.
    step_size (int): The step size between each window.

    Returns:
    tuple: A tuple containing:
        - X (2D array): A 2D array containing the combined features of the baseline and cognitive load datasets.
        - y (1D array): A 1D array containing the corresponding labels of the combined dataset.
        - feature_names (list): A list of feature names extracted from each window.
    """
    features_b, feature_names = create_windows_and_features(participant_id, 'baseline', window_size, step_size)
    features_c, _ = create_windows_and_features(participant_id, 'cognitive_load', window_size, step_size)
    
    uniform_length = min(len(features_b), len(features_c))      
    features_b = features_b[:uniform_length]
    features_c = features_c[:uniform_length]
    
    labels_b = np.zeros(len(features_b))  # Baseline = 0
    labels_c = np.ones(len(features_c))   # Cognitive Load = 1
    
    # Combine
    X = np.vstack([features_b, features_c])
    y = np.concatenate([labels_b, labels_c])
    
    return X, y, feature_names