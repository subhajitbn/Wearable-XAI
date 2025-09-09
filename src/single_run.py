"""
Main script for running the rule extraction pipeline on BVP signals.
This script loads baseline and cognitive load signals from CSV files, performs a grid search for optimal window and step sizes,
"""
# pylint: disable=wrong-import-order, wrong-import-position, ungrouped-imports
from suppress_errmsg import suppress_rpy2_and_other_errmsg, suppress_output
suppress_rpy2_and_other_errmsg()

# CLI tools
import click
from rich.console import Console
from rich.panel import Panel
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
from utils import load_signals_from_csv
from grid_search import grid_search_for_optimal_window_size
# from src.suppress_errmsg import silent_rpy2

# Regex for styling rules
import re
# pylint: enable=wrong-import-order, wrong-import-position, ungrouped-imports

# seed = 4242  # For reproducibility
console = Console()

def print_rules_pretty(rules):
    """
    Formats and prints a list of rules with stylized text using Rich library.

    Each rule is processed to extract and highlight different components:
    - Rule numbers are bolded and colored in magenta.
    - Numeric conditions and comparison operators are highlighted.
    - Numeric values and counts are colored in cyan and dimmed respectively.

    The formatted rules are then displayed in a panel with a green border.

    Parameters:
    rules (list of str): A list of rules to be formatted and printed.
    """

    styled_rules = []

    for rule in rules:
        # Extract rule number
        # number_match = re.match(r'\s*\[(\d+)\]', rule)
        # number = number_match.group(1) if number_match else "?"
        
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
    
def print_run_pipeline_outputs(optimal_window_size, optimal_step_size, rules, accuracy):
    """
    Prints the results of running the rule extraction pipeline.

    Parameters:
    optimal_window_size (int): The optimal window size for rule extraction.
    optimal_step_size (int): The optimal step size for rule extraction.
    rules (list of str): A list of extracted rules.
    accuracy (float): The accuracy of the extracted rules.
    """
    console.print(Rule("✅ Output"))
        
    panel = Panel.fit(
        Text("Optimal Window Size: " 
             + str(optimal_window_size) 
             + "\nOptimal Step Size: " 
             + str(optimal_step_size), style="bold white"),
        title="[bold green]Optimal Parameters",
        border_style="green",
        padding=(1, 2),
    )
    console.print(panel)

    # Display rules
    print_rules_pretty(rules)
    
    # Display accuracy    
    panel = Panel.fit(
        Text(f"Accuracy: {accuracy:.2f}", style="bold white"),
        title="[bold green]Prediction Metrics",
        border_style="green",
        padding=(1, 2),
    )
    console.print(panel)

# pylint: disable=too-many-locals, inconsistent-return-statements
def run_pipeline(baseline_path, cogload_path, seed, return_results):

    # Load signals
    """
    Executes the rule extraction pipeline on BVP signals, performing grid search for optimal window and step sizes,
    feature selection, and rule-based classification using SIRUS.

    Parameters:
    baseline_path (str): Path to the CSV file containing the baseline signal data.
    cogload_path (str): Path to the CSV file containing the cognitive load signal data.

    The function performs the following operations:
    1. Loads baseline and cognitive load signals from CSV files.
    2. Conducts a grid search to find optimal window and step sizes for feature extraction.
    3. Selects the top 3 features based on feature importances from a Random Forest model.
    4. Fits a SIRUS model on the selected features and extracts classification rules.
    5. Displays the extracted rules and calculates the model's prediction accuracy.

    The results are printed to the console using the Rich library for enhanced visualization.
    """

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
        # Push pandas DataFrame to R using context manager
        with localconverter(ro.default_converter + pandas2ri.converter):
            # Set seed in R environment
            if seed:
                ro.globalenv['seed'] = seed
            else:
                ro.globalenv['seed'] = 0
            ro.globalenv['features_for_sirus'] = ro.conversion.py2rpy(df)

        # R code block for SIRUS model fitting and rule extraction
        ro.r('''
        require(sirus)
        if(seed != 0) {
            set.seed(seed)  # For reproducibility
        }
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
    
    # Compute accuracy
    accuracy = np.mean(y_pred == y)
    
    if return_results:
        # Return everything for Streamlit
        return {
            "optimal_window_size": optimal_window_size,
            "optimal_step_size": optimal_step_size,
            "rules": rules,
            "accuracy": accuracy
        }
    
    print_run_pipeline_outputs(optimal_window_size, optimal_step_size, rules, accuracy)
        

# pylint: enable=too-many-locals, inconsistent-return-statements


@click.command()
@click.option("--baseline", required=True, type=click.Path(exists=True), help="CSV file for baseline signal.")
@click.option("--cogload", required=True, type=click.Path(exists=True), help="CSV file for cognitive load signal.")
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility. Defaults to None. Try with 4242.")
def cli(baseline, cogload, seed):
    """Run rule extraction pipeline on BVP signals."""
    return_results=False
    # with silent_rpy2():
    run_pipeline(baseline, cogload, seed, return_results)

# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    cli()
# pylint: enable=no-value-for-parameter
