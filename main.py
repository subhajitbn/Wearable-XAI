import click

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.rule import Rule

import numpy as np
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from src.utils import load_signals_from_csv, create_windows
from src.engineered_features import create_features
from src.grid_search import grid_search_for_optimal_window_size



def run_pipeline(baseline_path, cogload_path):

    # Load signals
    baseline_signal = load_signals_from_csv(baseline_path)
    cogload_signal = load_signals_from_csv(cogload_path)
    print("Signals loaded.")

    # A list of window sizes and step sizes for grid search
    window_sizes = [128, 256, 512, 1024, 2048]
    step_sizes = [64, 128, 256, 512, 1024]
    
    # Perform grid search for optimal window size and step size, and the corresponding model
    model, X, y, feature_names, optimal_window_size, optimal_step_size = grid_search_for_optimal_window_size(
        baseline_signal, cogload_signal, window_sizes, step_sizes
    )
    print(f"Optimal window size: {optimal_window_size}, Optimal step size: {optimal_step_size}")
    
    # Select top 3 features based on feature importances
    feature_importances = zip(feature_names, model.feature_importances_)
    sorted_feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    selected_features = [f for f, _ in sorted_feature_importances[:3]]
    # Create a DataFrame with selected features and labels
    df = pd.DataFrame(X, columns=feature_names).loc[:, selected_features].round(2)
    df["label"] = y

    # Push pandas DataFrame to R using context manager
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv['features_for_sirus'] = ro.conversion.py2rpy(df)

    # R code block for SIRUS model fitting and rule extraction
    ro.r('''
    require(sirus)

    data <- features_for_sirus[, -ncol(features_for_sirus)]
    data <- as.data.frame(data)
    y <- features_for_sirus$label

    sirus.m <- sirus.fit(data, y)
    rules <- capture.output(sirus.print(sirus.m))
    y_pred <- ifelse(sirus.predict(sirus.m, data) < 0.5, 0, 1)
    ''')

    # Pull back predictions and rules
    with localconverter(ro.default_converter + pandas2ri.converter):
        y_pred = np.array(ro.r('y_pred'), dtype=int)
        rules = list(ro.r('rules'))

    # Display rules
    console = Console()
    console.print(Rule("✅ Final Extracted Rules"))
    for idx, rule in enumerate(rules, start=1):
        console.print(f"[bold green]{idx}.[/bold green] {rule.strip()}")
    
    # Display accuracy
    accuracy = np.mean(y_pred == y)    
    console.print(Rule("✅ Prediction Accuracy"))
    console.print(f"[bold yellow]{accuracy:.2%}[/bold yellow]")

@click.command()
@click.option("--baseline", required=True, type=click.Path(exists=True), help="CSV file for baseline signal.")
@click.option("--cogload", required=True, type=click.Path(exists=True), help="CSV file for cognitive load signal.")
def cli(baseline, cogload):
    """Run rule extraction pipeline on BVP signals."""
    run_pipeline(baseline, cogload)

if __name__ == "__main__":
    cli()
