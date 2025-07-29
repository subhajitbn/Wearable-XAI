from src.suppress_errmsg import suppress_rpy2_and_other_errmsg, suppress_output
suppress_rpy2_and_other_errmsg()

# CLI tools
import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.rule import Rule

# Data science libraries
import numpy as np
import pandas as pd

# R integration via rpy2

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Custom modules
from src.utils import load_signals_from_csv, create_windows
from src.engineered_features import create_features
from src.grid_search import grid_search_for_optimal_window_size
# from src.suppress_errmsg import silent_rpy2

# Regex for styling rules
import re

seed = 4242  # For reproducibility
console = Console()

def print_rules_pretty(rules):
    styled_rules = []

    for rule in rules:
        # Extract rule number
        number_match = re.match(r'\s*\[(\d+)\]', rule)
        number = number_match.group(1) if number_match else "?"
        
        # Style condition
        rule_text = rule.strip()

        # Highlight numeric values and conditions
        rule_text = re.sub(r'([<>]=?|=)', r'[bold yellow]\1[/bold yellow]', rule_text)
        rule_text = re.sub(r'(\d+\.\d+|\d+)', r'[cyan]\1[/cyan]', rule_text)
        rule_text = re.sub(r'(n=\[?[\d]+\]?)', r'[dim]\1[/dim]', rule_text)

        # Bold the rule number
        rule_text = re.sub(r'^\s*\[(\d+)\]', r'[bold magenta][\1][/bold magenta]', rule_text)

        styled_rules.append(rule_text)

    panel = Panel.fit(
        "\n".join(styled_rules),
        title="[bold green]Extracted Rules",
        border_style="green",
        padding=(1, 2),
    )
    console.print(panel)


def run_pipeline(baseline_path, cogload_path):

    # Load signals
    baseline_signal = load_signals_from_csv(baseline_path)
    cogload_signal = load_signals_from_csv(cogload_path)

    # A list of window sizes and step sizes for grid search
    window_sizes = [128, 256, 512, 1024, 2048]
    step_sizes = [64, 128, 256, 512, 1024]
    
    # Perform grid search for optimal window size and step size, and the corresponding model
    model, X, y, feature_names, optimal_window_size, optimal_step_size = grid_search_for_optimal_window_size(
        baseline_signal, cogload_signal, window_sizes, step_sizes
    )
    
    # Select top 3 features based on feature importances
    feature_importances = zip(feature_names, model.feature_importances_)
    sorted_feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    selected_features = [f for f, _ in sorted_feature_importances[:3]]
    # Create a DataFrame with selected features and labels
    df = pd.DataFrame(X, columns=feature_names).loc[:, selected_features].round(2)
    df["label"] = y
    
    with suppress_output():
        # Set seed in R environment
        ro.r.assign("seed", seed)  
        
        # Push pandas DataFrame to R using context manager
        with localconverter(ro.default_converter + pandas2ri.converter):
            ro.globalenv['features_for_sirus'] = ro.conversion.py2rpy(df)

        # R code block for SIRUS model fitting and rule extraction
        ro.r('''
        require(sirus)
        set.seed(seed)  # For reproducibility
        
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
    console.print(Rule("✅ Output"))
    print_rules_pretty(rules)
    
    # Display accuracy
    accuracy = np.mean(y_pred == y)    
    panel = Panel.fit(
        Text(f"Accuracy: {accuracy:.2f}", style="bold white"),
        title="[bold green]Prediction Metrics",
        border_style="green",
        padding=(1, 2),
    )
    console.print(panel)
    
@click.command()
@click.option("--baseline", required=True, type=click.Path(exists=True), help="CSV file for baseline signal.")
@click.option("--cogload", required=True, type=click.Path(exists=True), help="CSV file for cognitive load signal.")
def cli(baseline, cogload):
    """Run rule extraction pipeline on BVP signals."""
    # with silent_rpy2():
    run_pipeline(baseline, cogload)

if __name__ == "__main__":
    cli()
