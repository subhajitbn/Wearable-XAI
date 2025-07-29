import os
import numpy as np
import pandas as pd

from .engineered_features import create_features

def load_signals_from_csv(file_path, column="bvp", sampling_rate=64, trim_seconds=5):
    """
    Load and preprocess a physiological signal from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file containing the physiological signal.
    column (str): Name of the column to extract. Default is "bvp".
    sampling_rate (int): Sampling rate of the signal in Hz. Default is 64 Hz.
    trim_seconds (int): Number of seconds to trim from start and end. Default is 5 seconds.

    Returns:
    np.ndarray: Trimmed physiological signal as a NumPy array.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")

    df = pd.read_csv(file_path)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in CSV. Available columns: {list(df.columns)}")

    signal = df[column].values

    if len(signal) < 2 * sampling_rate * trim_seconds:
        raise ValueError("Signal is too short to trim. Reduce trim_seconds or check the data.")

    start = sampling_rate * trim_seconds
    end = -start
    trimmed_signal = signal[start:end]

    return trimmed_signal


def create_windows(signal, window_size=256, step_size=64):
    """
    Create overlapping windows of given size from a signal.

    Parameters:
    signal (1D array): The signal to create windows from.
    window_size (int): The size of each window. Default is 256.
    step_size (int): The step size between each window. Default is 64.

    Returns:
    2D array: A 2D array containing the overlapping windows from the signal.
    """
    windows = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        window = signal[start:start+window_size]
        windows.append(window)
    return np.stack(windows)


def create_windows_and_features(signal, window_size, step_size):

    signal_windows = create_windows(signal, window_size = window_size, step_size = step_size)
    
    features = np.array([list(create_features(window).values()) for window in signal_windows])
    feature_names = list(create_features(signal_windows[0]).keys())
    
    return features, feature_names