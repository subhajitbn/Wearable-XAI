"""
Grid search for optimal window size and step size in time series data analysis.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from utils import create_windows_and_features

def make_uniform_length(baseline_features, cogload_features):
    """
    Ensure both feature sets are of uniform length by truncating to the minimum length.
    """
    uniform_length = min(len(baseline_features), len(cogload_features))
    return baseline_features[:uniform_length], cogload_features[:uniform_length]

def perform_random_forest_classification(X, y, seed):
    """
    Perform a random forest classification on the given data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    model : sklearn.ensemble.RandomForestClassifier
        Trained model.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    accuracy : float
        Accuracy of the model on the test set.
    """
    if seed:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, stratify=y, random_state=seed)
        model = RandomForestClassifier(n_estimators=100, random_state=seed)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, stratify=y)
        model = RandomForestClassifier(n_estimators=100)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    model_report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = model_report['accuracy']
    return model, X, y, accuracy

def grid_search_for_optimal_window_size(baseline_signal, cogload_signal, window_sizes, step_sizes, threshold=0.95, seed=None):
    """
    Perform a grid search for optimal window size and step size in time series data analysis.

    Parameters
    ----------
    baseline_signal : array-like of shape (n_samples,)
        Baseline signal.
    cogload_signal : array-like of shape (n_samples,)
        Cognitive load signal.
    window_sizes : list of int
        List of window sizes to try.
    step_sizes : list of int
        List of step sizes to try.
    threshold : float, optional
        Accuracy threshold for stopping the grid search. Defaults to 0.95.
    seed : int, optional
        Random seed for reproducibility. Defaults to 42.

    Returns
    -------
    model : sklearn.ensemble.RandomForestClassifier
        Trained model.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    feature_names : list of str
        Names of the features in X.
    window_size : int
        Optimal window size found.
    step_size : int
        Optimal step size found.
    """

    for window_size in window_sizes:
        for step_size in step_sizes:
            # Create windows and features for both baseline and cognitive load signals
            baseline_features, feature_names = create_windows_and_features(baseline_signal, window_size, step_size)
            cogload_features, _ = create_windows_and_features(cogload_signal, window_size, step_size)

            # Ensure both feature sets are of uniform length
            baseline_features, cogload_features = make_uniform_length(baseline_features, cogload_features)
            
            # Merge baseline and cognitive load features
            X = np.vstack([baseline_features, cogload_features])
            y = np.concatenate([np.zeros(len(baseline_features)), np.ones(len(cogload_features))])
            
            # Perform random forest classification 
            model, X, y, accuracy = perform_random_forest_classification(X, y, seed)
            
            if accuracy >= threshold:
                # print(f"Optimal window size found: {window_size} with step size {step_size}")
                # print("Classification Report:\n", classification_report(y_test, y_pred))
                return model, X, y, feature_names, window_size, step_size
            
    raise ValueError(f"Accuracy {accuracy} is below threshold.")
