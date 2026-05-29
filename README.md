# Corporate Bankruptcy Prediction — Italian Financial Challenge

**LUISS University · Academic Year 2025/2026 · Group 18**  
**Industry Partner: Expert.ai**

---

## Overview

This project builds a machine-learning system that reads Italian companies' annual financial statements and predicts whether each company will go bankrupt within the next twelve months. Rather than producing only a hard yes/no warning, the system also assigns every company a continuous 0–100 risk score and places it into one of six named risk tiers — from *Very Low* to *Very High* — so that credit analysts and risk managers can prioritize which firms deserve immediate attention without needing to review thousands of balance sheets manually.

The system was trained on four years of real Italian corporate data (2018–2021) and generates ranked predictions for the 2022–2023 period. Its primary business value is not the binary alarm but the ranked output: the model concentrates nearly all actual bankruptcies into the top 20% of its risk ranking, giving practitioners a focused watchlist rather than a flood of undifferentiated alerts.

---

## The Problem

Identifying companies at risk of bankruptcy before the event occurs is critical for banks, investors, and regulators — but it is structurally difficult. In Italy, fewer than 1% of companies fail in any given year, so a model that simply predicts "healthy" for everyone would be correct 99.3% of the time while being completely useless. The challenge requires a system that can find the rare signal of distress inside a large, imbalanced dataset, using only information that was available *before* the failure occurred, without accidentally incorporating future data into the model's training.

Three constraints shaped every design decision:

- **Extreme class imbalance**: ~0.7% of firm-years end in bankruptcy (~20 events per year)
- **Temporal structure**: training must use only historical data to predict future outcomes; random cross-validation would introduce leakage
- **Interpretability**: risk drivers must be traceable to real accounting signals, not black-box scores

---

## Dataset

| Attribute | Value |
|---|---|
| Source | Anonymized Italian corporate financial statements |
| Training period | 2018–2021 (4 fiscal years) |
| Test period | 2022–2023 (2 fiscal years) |
| Training observations | 11,828 company-year rows |
| Test observations | 5,811 company-year rows |
| Unique companies | 2,999 |
| Raw input features | 25+ (company identifiers, balance sheet items, income statement items, pre-calculated financial ratios) |
| Target variable | `bankruptcy_next_year` (binary: 0 = healthy, 1 = bankruptcy) |
| Bankruptcy rate (training) | ~0.7% per year (~63 total positive cases) |

Raw features include total assets, total debt (short- and long-term), shareholders' equity, production value, operating income, net profit/loss, and pre-calculated ratios (ROE, ROI, leverage, current ratio, quick ratio, debt-to-assets), alongside company metadata: ATECO sector code, province, region, legal form, and years in business.

No actual company data is included in this repository. All values are anonymized.

---

## What We Built

### Data Pipeline and Preprocessing

- Loaded and merged training and test CSVs; validated shapes, dtypes, and missing-value patterns
- Applied domain-specific missing-value imputation with documented justification per variable
- Winsorized extreme outliers at the 1st/99th percentile to reduce distortion in ratio variables
- Applied `log1p` transformation to right-skewed magnitude variables (total assets, total debt, production value, financial income)
- One-hot encoded categorical variables (ATECO sector, region, legal form, cluster ID) using the **development-data schema only**, then reindexed validation and test frames to the same columns — preventing any leakage from future categorical distributions

### Feature Engineering

All feature engineering is implemented in Section 10 of the notebook and produces a final matrix of ~79 model-ready features from the 25 raw inputs.

**Altman-like Z-Score**  
An adapted version of the Altman Z-score, recalibrated for the available variables (the original formulation requires retained earnings and market-value equity, which are absent here). The composite score summarises solvency, profitability, and asset-efficiency into a single distress indicator.

**Temporal trajectory features**  
Year-over-year growth rates, three-year rolling trends, and volatility estimates for key ratios (revenue, operating income, leverage, current ratio). These capture the *direction* of change, not just the current level.

**Distress trajectory features**  
Multi-year deterioration slopes for profit margin, leverage, current ratio, and debt-to-assets; consecutive-distress streak counters (how many years in a row a firm has been below a distress threshold); and second-order acceleration indicators (is the deterioration speeding up?).

**Sector-relative benchmarks**  
For each firm-year, key ratios (ROI, ROE, profit margin, leverage, debt-to-assets) are normalized against the sector-year median. This separates firm-specific weakness from sector-wide shocks and is the single most informative feature family in the final model.

**Two-stage unsupervised clustering**  
A broad 8-cluster K-Means pass (exploratory) feeds into a practical 4-cluster segmentation used as a model feature. The four clusters have distinct risk profiles:

| Cluster | Label | Share | Bankruptcy Rate |
|---|---|---|---|
| 0 | Distress Zone | 16% | 4.2% (~6× average) |
| 2 | Stable Majority | 36% | 0.09% |
| 1, 3 | Healthy Profiles | 48% | 0% observed |

Cluster 0 membership is the single strongest SHAP driver in the final model.

### Modelling Pipeline

**Temporal cross-validation**  
Expanding-window folds over the 2018–2020 development window (train on 2018 → validate on 2019; train on 2018–2019 → validate on 2020). The 2021 year is held entirely separate as the final model-selection holdout.

**Hyperparameter search**  
`RandomizedSearchCV` with `scoring='average_precision'` (PR-AUC, threshold-independent) for four model families:

| Model | Key imbalance handling |
|---|---|
| Logistic Regression | `class_weight` up to {0:1, 1:200} |
| Random Forest | `balanced_subsample` class weights |
| XGBoost | `scale_pos_weight` (~140× minority boost) |
| LightGBM | `class_weight='balanced'` |

SMOTE is applied **inside `imblearn.Pipeline`**, meaning it fires only on the training portion of each CV fold and is never applied to validation data.

**SMOTE ablation**  
For each model family, a parallel no-SMOTE variant is trained with identical hyperparameters and evaluated on the 2021 holdout. The empirically stronger configuration (with or without SMOTE) is selected per model before the final comparison. Results: SMOTE is marginally beneficial for Logistic Regression and LightGBM; neutral or slightly harmful for XGBoost and Random Forest.

**Threshold selection**  
Naive F1 maximisation on out-of-fold probabilities collapses to a ~0.01 cutoff that predicts almost every firm as bankrupt (recall → 1, precision → 0). To prevent this, the system selects a **constrained threshold**: F1 is maximised subject to `precision ≥ 10%`. This is computed per model on OOF probabilities, then frozen before the 2021 evaluation.

**Final evaluation**  
Each best-per-family pipeline is evaluated **exactly once** on the untouched 2021 validation year using the constrained threshold. LightGBM wins on F1. The winning pipeline is then refit on the full 2018–2021 labeled dataset before scoring the 2022–2023 test period.

### Neural Network

A fifth candidate model (Section 15): a **three-hidden-layer MLP** built in PyTorch.

| Layer | Units | Operations |
|---|---|---|
| Input | *n_features* | — |
| Hidden 1 | 256 | Linear → BatchNorm1d → ReLU → Dropout (35%) |
| Hidden 2 | 128 | Linear → BatchNorm1d → ReLU → Dropout (35%) |
| Hidden 3 | 64 | Linear → BatchNorm1d → ReLU → Dropout (35%) |
| Output | 1 | Linear (raw logit) |

Loss: `BCEWithLogitsLoss` with `pos_weight ≈ 140×`. Follows the same OOF threshold calibration and constrained-threshold logic as the tree-based models. Early stopping on validation PR-AUC prevents overfitting.

### Risk Scoring Output

Beyond the binary label, every firm in the 2022–2023 test set receives:

- A **raw bankruptcy probability** from the final LightGBM model
- A **0–100 risk score** (probability × 100, rounded to 2 decimal places)
- A **decile label** (D1 = safest 10%, D10 = riskiest 10%)
- A **named risk tier** mapped from deciles:

| Tier | Deciles | Interpretation |
|---|---|---|
| Very Low | D1–D2 | Bottom 20%, minimal concern |
| Low | D3–D4 | Below-average risk |
| Moderate | D5–D6 | Average-range risk |
| Elevated | D7–D8 | Above-average risk, worth monitoring |
| High | D9 | Second-highest risk band |
| Very High | D10 | Top 10%, priority review candidates |

581 firms in the 2022–2023 test period are assigned to the Very High tier.

### Interpretability

- **Feature importance** (native LightGBM split gain) — top 20 features reported
- **SHAP TreeExplainer** on the untouched 2021 validation set — directional impact of each feature on individual predictions
- **Error analysis** — top false negatives (missed bankruptcies) and false positives (false alarms) ranked by predicted probability, with diagnostics on why the model failed
- **Business threshold sensitivity study** — confusion-matrix counts and asymmetric cost functions (5× FN + 1× FP; 10× FN + 1× FP) computed across a threshold grid from 0.05 to 0.50

---

## Key Results

### Binary Classification — 2021 Validation Year

| Model | Threshold | Val F1 | Precision | Recall | ROC-AUC | PR-AUC | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|
| **LightGBM** *(selected)* | 0.31 | **0.124** | 0.079 | 0.286 | 0.918 | 0.054 | 6 | 70 | 15 | 2841 |
| Logistic Regression | 0.97 | 0.079 | 0.042 | 0.619 | 0.915 | 0.049 | 13 | 294 | 8 | 2617 |
| XGBoost | 0.36 | 0.078 | 0.054 | 0.143 | 0.937 | 0.063 | 3 | 53 | 18 | 2858 |
| Random Forest | 0.82 | 0.051 | 0.030 | 0.190 | 0.926 | 0.057 | 4 | 131 | 17 | 2780 |

*21 actual bankruptcies in the 2021 validation year. Neural network result computed at runtime; not hardcoded.*

### Risk Score — Decile Lift on 2021 Validation Year

| Decile | Firms | Bankruptcies | Bankruptcy Rate | Lift vs. Baseline |
|---|---|---|---|---|
| D1–D7 (combined) | ~2,060 | 1 | ~0.05% | ~0.07× |
| D8 | ~294 | 0 | 0.00% | 0× |
| D9 | ~294 | 7 | 2.4% | **3.3×** |
| D10 | ~294 | 13 | 4.4% | **6.2×** |
| **Total** | 2,932 | 21 | 0.72% | 1.0× (baseline) |

**What the lift means in practice:** If a credit analyst can only review 300 firms per year (roughly the top decile), using this model's ranking means they will encounter a bankrupt company roughly once every 23 reviews. Without any model, reviewing 300 randomly chosen firms from the same population would surface a bankruptcy roughly once every 143 reviews. The top-decile list is 6.2× more efficient than random selection. Extending the review to the top two deciles (D9 + D10, ~588 firms) captures 20 of 21 actual bankruptcies — 95% detection coverage at 20% of the portfolio.

### Cluster Risk Profile

| Cluster | Label | Share of firms | Bankruptcy rate | vs. average |
|---|---|---|---|---|
| 0 | Distress Zone | 16% | 4.2% | **6× above average** |
| 2 | Stable Majority | 36% | 0.09% | Below average |
| 1, 3 | Healthy Profiles | 48% | 0% observed | — |

Cluster 0 membership is the strongest single SHAP driver in the final model (mean absolute SHAP: highest among all features). The cluster requires only 9 input variables and can function as a standalone early-warning screen.

---

## Key Technical Decisions

### 1. Constrained threshold selection instead of naive F1 maximisation

When bankruptcies represent 0.7% of the data, the F1 loss surface has a pathological property: the global F1 maximum is often achieved at a threshold of 0.01–0.05, where the model predicts *everyone* as bankrupt. Recall approaches 1.0 but precision collapses to 0.7% (the base rate), and the resulting alarm system flags 99% of healthy companies for review.

The solution is to maximise F1 *subject to `precision ≥ 10%`*: at least 1 in 10 flagged firms must be a real bankruptcy. This forces the threshold search to operate in the range where the model is actually making a selective claim rather than a blanket one. The constraint value is intentionally modest — it acknowledges the dataset's structural difficulty while eliminating operationally absurd operating points. All results in this project use constrained thresholds.

### 2. Per-model SMOTE ablation rather than uniform application

SMOTE is a standard recommendation for imbalanced datasets, but it is not universally helpful. On tabular financial data with only ~20 positive training examples per fold, SMOTE generates synthetic minority-class observations by interpolating between real ones. For gradient-boosted trees (which learn threshold-like rules), synthetic observations close to real ones can actually *dilute* the distress signal rather than amplify it. The project tests this empirically: for each model family, a SMOTE and no-SMOTE variant are trained with identical hyperparameters and compared on the 2021 holdout. The winner is selected per model. This adds one evaluation round but prevents applying SMOTE where it degrades performance.

### 3. Sector-relative features as the primary risk signal

Raw financial ratios carry sector-wide variation that is not informative for bankruptcy prediction: a leverage ratio of 3.0 is unremarkable in utilities and alarming in retail. Normalising each firm's ratios against the ATECO sector-year median — constructing features like `profit_margin_vs_sector` and `leverage_vs_sector` — separates firm-specific deterioration from industry-wide conditions. This matters particularly in a dataset spanning COVID-19 (2020) and the energy crisis (2021-2022), which introduced large sector-wide shocks. The SHAP analysis confirms that sector-relative features are the two most important predictors in the final model, ahead of any absolute ratio.

---

## Repository Structure

```
italian-financial-challenge/
│
├── GrandChallenge_1,_Group_18.ipynb   Main analysis notebook (Sections 1–15)
├── README.md                          This file
├── requirements.txt                   Python dependencies
│
├── data/
│   └── processed/
│       ├── train_data.csv             Training set: 11,828 rows, 2018–2021
│       ├── test_features.csv          Test features: 5,811 rows, 2022–2023 (no labels)
│       ├── challenge1_test_predictions.csv     Binary 0/1 predictions (generated)
│       └── challenge1_test_risk_scores.csv     Risk score + decile + tier (generated)
│
├── docs/
│   ├── challenge_description.md       Full challenge specification and evaluation criteria
│   └── data_dictionary.md             Variable definitions and data quality notes
│
└── notebooks/
    └── starter_template.ipynb         Original course-provided template (reference only)
```

The two files under `data/processed/` marked *generated* are produced when the notebook is run end-to-end and are not committed to the repository.

---

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd italian-financial-challenge

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
# Open GrandChallenge_1,_Group_18.ipynb and run all cells in order
```

> **PyTorch note:** Section 15 (Neural Network) requires `torch`. If you only want to run Sections 1–14, PyTorch is not needed. Install with `pip install torch` or follow the [official instructions](https://pytorch.org/get-started/locally/) for your platform and CUDA version.

---

## Stack

| Category | Library / Tool |
|---|---|
| Language | Python 3 |
| Data manipulation | pandas, numpy |
| Statistics | scipy |
| Visualisation | matplotlib, seaborn, plotly |
| Machine learning | scikit-learn |
| Gradient boosting | XGBoost, LightGBM |
| Imbalance handling | imbalanced-learn (SMOTE, ImbPipeline) |
| Neural networks | PyTorch |
| Model interpretability | SHAP |
| Notebook environment | Jupyter Notebook |

Full version constraints are specified in `requirements.txt`.

---

## Academic Context

This project was completed as part of the **Italian Financial Data Challenge**, a course assignment at **LUISS University** (Academic Year 2025/2026) run in collaboration with **Expert.ai**. The work was carried out by Group 18: Sofia Capriolo, Andrea Cipolla, Giorgio Vanini, and Arianna Cambi. All data is anonymized and used strictly for educational purposes; nothing in this repository should be interpreted as investment, credit, or financial advice. External code and resources are cited within the notebook cells where applicable.
