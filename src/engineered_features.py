import numpy as np
from scipy.signal import find_peaks, peak_widths, peak_prominences

def max_amp(window):
    """
    Calculate the maximum amplitude of a signal window.

    Parameters
    ----------
    window : ndarray
        Signal window.

    Returns
    -------
    amplitude : float
        Maximum amplitude of the signal window.

    Notes
    -----
    The amplitude is calculated as the difference between the maximum and minimum
    values in the window, divided by 2.
    """
    return (window.max() - window.min()) / 2

def sign_changes(window):
    """
    Calculate the number of sign changes in the derivative of a signal window.

    Parameters
    ----------
    window : ndarray
        Signal window.

    Returns
    -------
    num_sign_changes : int
        Number of sign changes in the derivative of the signal window.

    Notes
    -----
    The derivative is calculated using the difference function, and the sign is
    calculated using the sign function. The number of sign changes is then
    calculated as the number of times the sign changes from one element to the
    next.
    """
    derivative = np.diff(window)
    return np.sum(np.diff(np.sign(derivative)) != 0)

def mean_peak_width(signal):
    """
    Calculate the mean peak width of a signal.

    Parameters
    ----------
    signal : ndarray
        Signal.

    Returns
    -------
    mean_width : float
        Mean width of the peaks in the signal.

    Notes
    -----
    The width is calculated as the width of the peak at half of its maximum
    value, relative to the baseline of the peak. The mean width is then
    calculated as the mean of the widths of all the peaks in the signal.
    """
    peaks, _ = find_peaks(signal, prominence=1)  # Adjust prominence as needed
    if len(peaks) == 0:
        return 0
    widths, _, _, _ = peak_widths(signal, peaks, rel_height=0.5)
    return np.mean(widths)

def mean_peak_prominence(signal):
    """
    Calculate the mean prominence of peaks in a signal.

    Parameters
    ----------
    signal : ndarray
        Signal data.

    Returns
    -------
    mean_prominence : float
        Mean prominence of the peaks in the signal.

    Notes
    -----
    Prominence is a measure of how much a peak stands out from the surrounding
    baseline of the signal. The mean prominence is calculated as the average
    prominence of all detected peaks in the signal.
    """

    peaks, _ = find_peaks(signal, prominence=1)  # Adjust prominence as needed
    if len(peaks) == 0:
        return 0
    prominences = peak_prominences(signal, peaks)[0]
    return np.mean(prominences)

def post_peak_area(signal, time_window=50):
    """
    Calculate the area under the signal after the main peak within a specified time window.

    Parameters
    ----------
    signal : ndarray
        The input signal data.
    time_window : int, optional
        Number of data points to include in the area calculation after the main peak (default is 50).

    Returns
    -------
    area : float
        The sum of absolute values of the signal data points after the main peak within the specified time window.

    Notes
    -----
    The function identifies peaks in the signal and calculates the area under the signal starting from
    the main peak (highest peak) up to the specified time window. If no peaks are found, the function returns 0.
    """

    peaks, _ = find_peaks(signal)
    if len(peaks) == 0:
        return 0
    main_peak_idx = peaks[np.argmax(signal[peaks])]
    right_end = min(len(signal), main_peak_idx + time_window)
    return np.sum(np.abs(signal[main_peak_idx:right_end]))

def pre_peak_area(signal, time_window=50):
    """
    Calculate the area under the signal before the main peak within a specified time window.

    Parameters
    ----------
    signal : ndarray
        The input signal data.
    time_window : int, optional
        Number of data points to include in the area calculation before the main peak (default is 50).

    Returns
    -------
    area : float
        The sum of absolute values of the signal data points before the main peak within the specified time window.

    Notes
    -----
    The function identifies peaks in the signal and calculates the area under the signal up to the main peak
    (highest peak) within the specified time window. If no peaks are found, the function returns 0.
    """
    peaks, _ = find_peaks(signal)
    if len(peaks) == 0:
        return 0
    main_peak_idx = peaks[np.argmax(signal[peaks])]
    left_start = max(0, main_peak_idx - time_window)
    return np.sum(np.abs(signal[left_start:main_peak_idx]))

def create_features(window):
    """
    Calculate a set of features from a signal window.

    Parameters
    ----------
    window : ndarray
        Signal window.

    Returns
    -------
    features : dict
        A dictionary containing the following features:

        - max_amp: maximum amplitude of the signal
        - sign_changes: number of sign changes in the derivative of the signal
        - mean_peak_width: mean width of the peaks in the signal
        - mean_peak_prominence: mean prominence of the peaks in the signal
        - post_peak_area: area under the signal after the main peak
        - pre_peak_area: area under the signal before the main peak
    """
    return {
        'max_amp': max_amp(window),
        'sign_changes': sign_changes(window),
        'mean_peak_width': mean_peak_width(window),
        'mean_peak_prominence': mean_peak_prominence(window),
        'post_peak_area': post_peak_area(window),
        'pre_peak_area': pre_peak_area(window)
    }
    
