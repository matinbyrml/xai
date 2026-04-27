# Explainable AI (XAI) Portfolio

## Overview
This repository contains a collection of Jupyter Notebooks demonstrating practical implementations of Explainable Artificial Intelligence (XAI) techniques. The notebooks progress from inherently interpretable white-box models to advanced post-hoc techniques for explaining complex black-box models (like Deep Random Forests and Gradient Boosted machines).

The primary datasets used across these modules are the **Titanic Survival Dataset** and the **Adult Census Income Dataset**, providing consistent baselines to compare how different XAI methods interpret the exact same features.

## Repository Structure

### Interpretable by Design (White-Box Models)
*Focus: Building models that are transparent and mathematically interpretable from the ground up.*
* **Decision Trees:** Extracting global feature importance and tracing local rule-based decision paths.
* **Logistic Regression:** Interpreting model weights as Odds Ratios (and managing the pitfalls of multicollinearity).
* **K-Nearest Neighbors (KNN):** Utilizing instance-based learning for example-based local explanations.

### Global Post-Hoc Explainability
*Focus: Opening the "Black Box" to understand the global rules and behaviors learned by complex, uninterpretable models across an entire dataset.*
* **Permutation Feature Importance:** Using the "Shuffle Test" to identify true feature reliance and detect model overfitting.
* **Partial Dependence Plots (PDP):** Visualizing the marginal effect of individual features (e.g., Age, Fare) on predicted outcomes.
* **Global Surrogate Models:** Training transparent white-box models to mimic and explain the logic of a complex `GradientBoostingClassifier`, measured via Fidelity and Reconstruction Error.

### Local Post-Hoc Explainability (Linear Surrogates)
*Focus: Zooming in to explain the model's exact reasoning for a single, specific instance using linear approximations.*
* **LIME (Local Interpretable Model-agnostic Explanations):** * Building local linear surrogates via data perturbation and proximity weighting.
  * Handling complex dual-path preprocessing (Label Encoding vs. One-Hot Encoding) to prevent mathematically impossible perturbations.
  * Analyzing the stability of local explanations and the impact of hyperparameters like `num_samples` and distance metrics.

### Local Post-Hoc Explainability (Game Theory)
*Focus: Applying cooperative game theory to guarantee exact feature attribution for individual predictions.*
* **SHAP (Shapley Additive Explanations):**
  * Building background datasets (Maskers) to establish baseline expected values (`E[f(x)]`).
  * Visualizing local feature contributions using interactive Force Plots and modern Waterfall Plots.
  * Aggregating local SHAP values to extract global insights via Summary plots (Bee Swarm) and Dependence plots (with feature interaction coloring).

## Tech Stack & Libraries
* **Language:** Python 3.x
* **Core Data Science:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (Decision Trees, Random Forests, Gradient Boosting, SVM, Logistic Regression)
* **XAI Libraries:** `lime`, `shap`, `sklearn.inspection` (Permutation, PDP)
* **Visualization:** `matplotlib`, `seaborn`

## Key Concepts Mastered
Throughout this portfolio, several core XAI theoretical concepts are practically demonstrated:
1. **The XAI Taxonomy:** Global vs. Local scope, Model-Agnostic vs. Model-Specific techniques.
2. **The Accuracy-Interpretability Trade-off:** Balancing model complexity with human readability.
3. **Data Leakage & Preprocessing:** Ensuring robust pipelines (splitting before scaling/imputing) to maintain explainer integrity.
4. **Explanation Stability & Exactness:** Understanding the fragility of non-deterministic explainers (like LIME) versus the exact, additive guarantees of game-theoretic explainers (like SHAP).

## How to Use
1. Clone this repository.
2. Ensure you have the required libraries installed: `pip install scikit-learn pandas numpy matplotlib seaborn lime shap`
3. Launch Jupyter Notebook or Jupyter Lab and open the `.ipynb` files to execute the cells sequentially.
