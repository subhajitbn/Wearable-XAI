# Personalized Wearables via Rule-Based eXplainable AI

## 🚀 Motivation

Wearable devices collect deeply personal physiological data, yet their core inference models are often opaque, centralized, and poorly suited to individual needs. Over-reliance on black-box AI models trained on unrepresentative datasets risks both bias and misinterpretation.

Our goal is to invert this paradigm: use AI not as a decision-maker, but as a *discovery tool*—to extract interpretable, human-aligned models that can be deployed on-device and easily personalized.

This repository demonstrates a concrete proof-of-concept for that philosophy, using blood volume pulse (BVP) signals from the [CogWear](https://doi.org/10.13026/5f6t-b637) dataset to model cognitive load via rule extraction from machine learning.

---

## Dependencies

## 📦 Install Dependencies
Note that [R]() is a dependency of this project.
First, make sure you have [uv](https://github.com/astral-sh/uv) installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the root of the repository:
```bash
uv venv         # Create a virtual environment
uv pip install -e .  # Install your package in editable mode from pyproject.toml
```



---

## 📂 Repository Structure

### `exploration.ipynb`

An interactive, step-by-step notebook to:
- Visualize BVP signals from a single participant
- Propose and compute six meaningful engineered features
- Train a Random Forest classifier
- Analyze feature importance using MDI (Mean Decrease in Impurity)
- Extract rules using **SIRUS**
- Evaluate classification performance and interpret results

> This notebook captures the reasoning process—what features matter, why they matter, and how the model can be simplified.

---

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

This repo is not about pushing model accuracy. Indeed, it's hard to define the true label of a BVP window. A person can be quite calm at times while handling cognitive load, and somewhat agitated at times while providing baseline measurements. It's about showing how we can **bootstrap explainable, symbolic models from machine learning**, refine them with **human judgment**, and **deploy interpretable logic** instead of opaque AI. We believe this is the future of trustworthy, personalizable AI in wearables.