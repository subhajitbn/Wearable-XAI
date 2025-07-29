import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from .utils import create_windows_and_features

def grid_search_for_optimal_window_size(baseline_signal, cogload_signal, window_sizes, step_sizes, threshold=0.95, seed=42):
    for window_size in window_sizes:
        for step_size in step_sizes:
            # Create windows and features for both baseline and cognitive load signals
            baseline_features, feature_names = create_windows_and_features(baseline_signal, window_size, step_size)
            cogload_features, _ = create_windows_and_features(cogload_signal, window_size, step_size)

            # Ensure both feature sets are of uniform length
            uniform_length = min(len(baseline_features), len(cogload_features))
            baseline_features = baseline_features[:uniform_length]
            cogload_features = cogload_features[:uniform_length]
            
            # Merge baseline and cognitive load features
            X = np.vstack([baseline_features, cogload_features])
            y = np.concatenate([np.zeros(len(baseline_features)), np.ones(len(cogload_features))])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, stratify=y, random_state=seed)
            model = RandomForestClassifier(n_estimators=100, random_state=seed)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            model_report = classification_report(y_test, y_pred, output_dict=True)
            accuracy = model_report['accuracy']
            
            if accuracy >= threshold:
                # print(f"Optimal window size found: {window_size} with step size {step_size}")
                # print("Classification Report:\n", classification_report(y_test, y_pred))
                return model, X, y, feature_names, window_size, step_size
            else:
                raise ValueError(f"Accuracy {accuracy} is below threshold.")