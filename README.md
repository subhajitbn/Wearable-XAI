# Personalized Wearables via Rule-Based eXplainable AI

## 💡 Motivation

Wearable devices collect deeply personal physiological data, yet their core inference models are often opaque, centralized, and poorly suited to individual needs. Over-reliance on black-box AI models trained on unrepresentative datasets risks both bias and misinterpretation.

Our goal is to invert this paradigm: use AI not as a decision-maker, but as a *discovery tool*—to extract interpretable, human-aligned models that can be deployed on-device and easily personalized.

This repository demonstrates a concrete proof-of-concept for that philosophy, using blood volume pulse (BVP) signals from the [CogWear](https://doi.org/10.13026/5f6t-b637) dataset to model cognitive load via rule extraction from machine learning.

---

## 🚀 Option 1: Run in a Docker Container

This is the easiest and cleanest way to get started. You do **not** need to install R, Python dependencies, or `uv` locally.

### Build the Docker image:
From the root of the repository:
```bash
docker build -t wearable-xai .
```

### Run the CLI in the container:
To run the CLI with already provided CSV files for baseline and cognitive load signals of participant 3 from the CogWear dataset:
```bash
docker run --rm -v $PWD:/app wearable-xai uv run main.py --baseline baseline_bvp_p3.csv --cogload cogload_bvp_p3.csv --seed 4242
```

### Run the web app in the container:
```bash
docker run --rm -p 8501:8501 --rm -v $PWD:/app wearable-xai uv run streamlit run app.py
```
or, simply,
```bash
docker run -p 8501:8501 wearable-xai
```

---

## 🧪 Option 2: Run Locally without Container

### Install dependencies
The project depends on [R](https://www.r-project.org) (4.3.1) and [uv](https://github.com/astral-sh/uv), which can be installed from their respective websites. 

Next, We need the `SIRUS` package from CRAN. This is the main eXplainable AI package that we use for rule extraction. For details, see the original articles [here](https://proceedings.mlr.press/v130/benard21a.html) and [here](https://doi.org/10.1214/20-EJS1792). We also refer to the excellent [Implementation Overview](https://sirus.jl.huijzer.xyz/dev/implementation-overview/) section in the documentation of [SIRUS.jl](https://github.com/rikhuijzer/SIRUS.jl).
```bash
Rscript -e "install.packages('sirus', repos='https://cloud.r-project.org')"
```
After installing R, `SIRUS` and uv, run the following command to sync Python dependencies:
```bash
uv sync --locked
```

### Run the CLI
To run the CLI with already provided CSV files for baseline and cognitive load signals of participant 3 from the CogWear dataset:
```bash
uv run main.py --baseline baseline_bvp_p3.csv --cogload cogload_bvp_p3.csv --seed 4242
```

### Run the web app
```bash
uv run streamlit run app.py
```

---

## 📘 Pipeline Breakdown: See the Two Notebooks

### `exploration.ipynb`

An interactive, step-by-step notebook to:
- Visualize BVP signals from a single participant
- Propose and compute six meaningful engineered features
- Train a Random Forest classifier
- Analyze feature importance using MDI (Mean Decrease in Impurity)
- Extract rules using **SIRUS**
- Evaluate classification performance and interpret results

> This notebook captures the reasoning process—what features matter, why they matter, and how the model can be simplified.


### `pipeline.ipynb`

An end-to-end, semi-automated pipeline that:
- Discovers optimal window size and step size for segmentation
- Computes features from each window
- Trains a Random Forest model
- Uses `rpy2` to interface with the R implementation of **SIRUS**
- Extracts rules for a compact, deployable model.

> This notebook reflects the production mindset—automated preprocessing and rule extraction for scalable deployment.

---

## 🔍 Summary

This project is not about pushing model accuracy. Indeed, it's hard to define the true label of a BVP window. A person can be quite calm at times while handling cognitive load, and somewhat agitated at times while providing baseline measurements. It's about showing how we can **bootstrap explainable, symbolic models from machine learning**, refine them with **human judgment**, and **deploy interpretable logic** instead of opaque AI. We believe this is the future of trustworthy, personalizable AI in wearables.