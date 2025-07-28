import numpy as np
from scipy.signal import find_peaks, peak_widths, peak_prominences

def max_amp(window):
    return (window.max() - window.min()) / 2

def sign_changes(window):
    derivative = np.diff(window)
    return np.sum(np.diff(np.sign(derivative)) != 0)

def mean_peak_width(signal):
    peaks, _ = find_peaks(signal, prominence=1)  # Adjust prominence as needed
    if len(peaks) == 0:
        return 0
    widths, _, _, _ = peak_widths(signal, peaks, rel_height=0.5)
    return np.mean(widths)

def mean_peak_prominence(signal):
    peaks, _ = find_peaks(signal, prominence=1)  # Adjust prominence as needed
    if len(peaks) == 0:
        return 0
    prominences = peak_prominences(signal, peaks)[0]
    return np.mean(prominences)

def post_peak_area(signal, time_window=50):
    peaks, _ = find_peaks(signal)
    if len(peaks) == 0:
        return 0
    main_peak_idx = peaks[np.argmax(signal[peaks])]
    right_end = min(len(signal), main_peak_idx + time_window)
    return np.sum(np.abs(signal[main_peak_idx:right_end]))

def pre_peak_area(signal, time_window=50):
    peaks, _ = find_peaks(signal)
    if len(peaks) == 0:
        return 0
    main_peak_idx = peaks[np.argmax(signal[peaks])]
    left_start = max(0, main_peak_idx - time_window)
    return np.sum(np.abs(signal[left_start:main_peak_idx]))

def create_features(window):
    return {
        'max_amp': max_amp(window),
        'sign_changes': sign_changes(window),
        'mean_peak_width': mean_peak_width(window),
        'mean_peak_prominence': mean_peak_prominence(window),
        'post_peak_area': post_peak_area(window),
        'pre_peak_area': pre_peak_area(window)
    }
    
