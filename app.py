"""
Streamlit app for rule extraction and visualization.
"""
import os
import re
import tempfile

import streamlit as st

from single_run import run_pipeline



def format_rule_latex(rule: str) -> str:
    """
    Converts a single rule into LaTeX format.
    - Replaces operators with LaTeX equivalents
    - Escapes underscores
    - Wraps numbers and symbols appropriately
    """
    rule = rule.strip()

    # Escape underscores
    rule = rule.replace("_", r"\_")

    # Fix logical AND (replace & with \land)
    rule = rule.replace("&", r"\land")

    # Bold rule ID
    rule = re.sub(r'^\s*\[(\d+)\]', lambda m: f'[\\textbf{{{int(m.group(1)) - 1}}}]', rule)

    # Add LaTeX text commands for logical flow with color
    rule = re.sub(r'\bif\b', r'\\;\\textcolor{orange}{\\text{If}}\\quad', rule)
    rule = re.sub(r'\bthen\b', r'\\quad\\textcolor{green}{\\text{then}}\\quad', rule)
    rule = re.sub(r'\belse\b', r'\\quad\\textcolor{red}{\\text{else}}\\quad', rule)


    # Inequalities
    rule = rule.replace(">=", r"\geq").replace("<=", r"\leq")

    # Sample size styling
    rule = re.sub(r'\(n=\[?(\d+)\]?\)', r'\\;\\textcolor{gray}{(n=\1)}', rule)
    

    # Remove quotations
    rule = rule.replace('"', '')

    return f"{rule}"


def render_rules_latex(rules: list[str]):
    """Render all rules as a single left-aligned LaTeX block in Streamlit."""
    # Original first rule
    first_line = rules[0].strip()

    # Remove any [index] prefix
    first_line = re.sub(r'^\s*\[\d+\]\s*', '', first_line)

    # Remove all leading/trailing quotes — straight and curly
    first_line = first_line.strip('"“”\'')
    st.markdown(f"**{first_line}**")

    
    formatted_lines = [format_rule_latex(rule) + r" \\" for rule in rules[1:]]
    latex_block = r"""
    \begin{array}{l}
    """ + "\n".join(formatted_lines) + r"""
    \end{array}
    """
    st.latex(latex_block)



st.set_page_config(page_title="Wearable-XAI Rule Extractor for Cognitive Load", layout="centered")

st.title("Wearable-XAI")
st.subheader("Extract rules from BVP (blood volume pulse) signals to predict cognitive load 🧠")
st.markdown("Upload two CSV files of BVP signals — one for **baseline** and one for **cognitive load**.")

baseline_file = st.file_uploader("Upload Baseline CSV", type=["csv"])
cogload_file = st.file_uploader("Upload Cognitive Load CSV", type=["csv"])
seed = st.number_input("Seed", value=4242, step=1)

if st.button("Run Extraction"):
    if not (baseline_file and cogload_file):
        st.error("Please upload both baseline and cognitive load CSV files.")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = os.path.join(tmpdir, "baseline.csv")
            cogload_path = os.path.join(tmpdir, "cogload.csv")
            with open(baseline_path, "wb") as f:
                f.write(baseline_file.read())
            with open(cogload_path, "wb") as f:
                f.write(cogload_file.read())

            results = run_pipeline(baseline_path, cogload_path, seed, return_results=True)

            st.success("Pipeline executed successfully!")

            st.subheader("🔧 Optimal Parameters")
            st.markdown(f"**Window Size:** {results['optimal_window_size']}")
            st.markdown(f"**Step Size:** {results['optimal_step_size']}")

            st.subheader("📜 Extracted Rules")
            render_rules_latex(results['rules'])

            st.subheader("📊 Performance Metrics")
            st.markdown(f"**Accuracy:** {results['accuracy']:.2%}")
