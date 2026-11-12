# End-to-End Credit Risk & Default Prediction Framework
### *Benchmarking Classical Baselines, Ensembles, and Neural Networks with Explainable AI (SHAP)*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Champion_Model-LightGBM-success.svg)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-TreeSHAP-orange.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary

This repository presents an industry-grade **Credit Risk Modeling & Default Prediction Framework** built on a synthetic 50,000-borrower financial portfolio with real-world statistical properties (heavy Pareto tails, zero-inflated delinquency counts, and complex missing mechanisms). 

Addressing the standard banking challenge of class imbalance (~11% default rate), this project demonstrates:
1. **Empirical Missing Data Auditing:** Diagnosing **MCAR** vs. **MAR** mechanisms and designing a leakage-free imputation architecture.
2. **Hybrid Column Transformation:** Selectively pairing non-linear power transformations (`Yeo-Johnson`) for skewed financial amounts with `RobustScaler` and `OneHotEncoder`.
3. **5-Model Cross-Validation Tournament:** Evaluating **Logistic Regression**, **Random Forest**, **Multi-Layer Perceptron (MLP)**, **XGBoost**, and **LightGBM** across 5-fold stratified splits, prioritizing **PR-AUC (Precision-Recall AUC)** over the standard ROC-AUC illusion.
4. **Explainable AI (TreeSHAP) & Regulatory Compliance:** Transforming the winning **LightGBM** model into fully transparent, legally compliant **Adverse Action Notices** (FCRA/ECOA) and performing **Confusion Matrix Error Slicing** (auditing why false negatives slipped past the model).

---

## 🏗️ Project Architecture

```
[ 1. Realistic Synthesis ] ──► [ 2. Missingness Audit ] ──► [ 3. Leakage-Free Preprocessing ]
  • Pareto/Log-normal income     • Credit Score: MCAR        • Stratified Train/Test Split (80/20)
  • Zero-inflated delinquencies  • Employment Length: MAR    • Age-bracket median imputation
  • Imbalanced defaults (~11%)     (50% spike at 18-24)      • Hybrid ColumnTransformer

                                          │
                                          ▼
[ 6. Explainable AI & Audit ] ◄── [ 5. Model Tournament ] ◄── [ 4. Metric Selection ]
  • Global Beeswarm Plot          • 5-Fold Stratified CV      • The "ROC-AUC Trap"
  • Unscaled Waterfall Reports    • Baseline vs. Challengers  • Prioritizing PR-AUC for
  • Confusion Matrix Slicing      • LightGBM Champion (0.639)   minority default detection
```

---

## 📊 1. The Dataset & Missing Mechanism Diagnosis

Financial data in production is never normally distributed. The dataset mimics real-world banking characteristics:
* **`annual_income`:** Log-normal distribution with fat Pareto tails (0.5% extreme earners up to \$3,000,000).
* **`dti_ratio`:** Gamma-distributed with severe distress outliers (> 0.8 up to 2.5).
* **`revolving_utilization`:** Beta-distributed with over-limit cardholders (> 100%).

### Missingness Audit (Empirical Proof of MAR vs. MCAR)
Before imputing, we audited missing rates across demographic quintiles:
* **`credit_score`:** Consistently flat (~3.5%) across all age, income, and housing brackets $\rightarrow$ **MCAR (Missing Completely at Random)**.
* **`employment_length_years`:** Exhibited a massive spike to **~50% in the 18–24 age cohort** (students and entry-level workers) and flat 5% elsewhere $\rightarrow$ **MAR (Missing at Random) conditioned on Age**.

---

## ⚙️ 2. Preprocessing & Leakage Prevention Pipeline

1. **Strict Partitioning First:** Split into Train (80%) and Holdout Test (20%) using stratified sampling before calculating any transformation parameters.
2. **Context-Aware Imputation:**
   * `employment_length_years`: Imputed using the **median of the applicant's specific age cohort** learned exclusively from the training set (preventing an 18-year-old from being assigned 6 years of work experience).
   * `credit_score`: Imputed with the global training median.
   * **Missingness Indicators:** Created binary flags (`credit_score_was_missing`, `emp_len_was_missing`) to preserve the credit bureau "thin-file" risk signal.
3. **Hybrid `ColumnTransformer` Architecture:**
   * **`PowerTransformer(method='yeo-johnson')`:** Applied to heavily skewed dollar amounts (`annual_income`, `loan_amount`, `dti_ratio`, `avg_tx_amount_30d`) to normalize variance and linearize log-odds.
   * **`RobustScaler()`:** Applied to bounded continuous features (`credit_score`, `open_credit_lines`, `revolving_utilization`) using median and IQR to neutralize extreme outlier distortion.
   * **`OneHotEncoder(drop='first')`:** Applied to nominal categories (`home_ownership`, `loan_intent`).
   * **`passthrough`:** Preserved binary flags as pure 0/1 inputs.

---

## 🏆 3. Model Benchmark & Evaluation Tournament

Because defaults represent only ~11% of applicants, **Accuracy is meaningless** and **ROC-AUC is deceptive** (the massive True Negative count masks false alarms). We established **PR-AUC (Precision-Recall AUC)** as our primary objective metric, run under **5-Fold Stratified Cross-Validation**:

| Rank | Model Paradigm | CV PR-AUC (Mean) | CV PR-AUC (Std) | CV ROC-AUC (Mean) | Test PR-AUC | Test ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **3. LightGBM (Leaf-Wise Boosting)** | **0.6315** | $\pm 0.0214$ | **0.8772** | **0.6395** | **0.8787** |
| 🥈 | **2. XGBoost (Level-Wise Boosting)** | 0.6301 | $\pm 0.0219$ | 0.8764 | 0.6375 | 0.8791 |
| 🥉 | **4. Neural Network (MLP: 64-32)** | 0.6241 | $\pm 0.0192$ | 0.8764 | 0.6318 | 0.8739 |
| 4 | **1. Logistic Regression (Baseline)** | 0.6060 | $\pm 0.0231$ | 0.8700 | 0.6062 | 0.8695 |
| 5 | **5. Random Forest (Bagging)** | 0.5905 | $\pm 0.0183$ | 0.8728 | 0.5975 | 0.8731 |

### Key Benchmark Findings:
* **The ROC-AUC Illusion Proven:** All 5 models scored virtually identical ROC-AUC scores (~0.870 to 0.879), while PR-AUC exposed a massive **4.9 percentage point performance gap** between Random Forest and LightGBM.
* **Flawless Generalization:** Every model’s Cross-Validation PR-AUC matched its Holdout Test score within $\pm 0.008$, proving that the preprocessing pipeline suffered **zero data leakage**.
* **Gradient Boosting Dominance:** Sequential residual learning (LightGBM/XGBoost) significantly outperformed parallel tree averaging (Random Forest) in isolating minority default risk.

---

## 🔍 4. Explainable AI (TreeSHAP) & Banking Compliance

Modern banking regulations (FCRA, ECOA, SR 11-7) strictly forbid deploying opaque "black-box" models. We integrated **TreeSHAP** with the champion LightGBM model:

### A. Global Risk Drivers (Beeswarm Analysis)
Over 70% of the model's total decision weight is driven by four core fundamentals:
1. `credit_score` (Inversely correlated with risk)
2. `annual_income` (Protective against cash-flow shocks)
3. `dti_ratio` (Strong positive risk multiplier)
4. `revolving_utilization` (Direct indicator of credit line distress)

### B. Local Adverse Action Notices (Unscaled Waterfall Explanations)
By mapping SHAP values back to **original real-world units** (dollars, FICO points, actual DTI), the framework automatically produces legally compliant rejection reasons:
* *Applicant Example:* Declined with **72.0% Predicted Default Probability**.
  * Primary Reason: `dti_ratio = 0.85` (+32.5% risk contribution)
  * Secondary Reason: `revolving_utilization = 1.05` (+21.0% risk contribution)
  * Tertiary Reason: `credit_score = 540` (+18.4% risk contribution)

---

## 🔬 5. Confusion Matrix Error Slicing (Auditing Model Failures)

By breaking down mean feature impact across the four confusion matrix quadrants, we audited the root causes of model accuracy and failure:

```
=== PERCENTAGE SHARE OF TOTAL DECISION IMPACT (%) ===
Quadrant                          credit_score   annual_income   dti_ratio   loan_amount   revolving_util   All Other
TP: Caught Defaulter                 35.8%           15.1%         12.1%        10.4%           8.3%          18.3%
FP: False Alarm (Lost Customer)      27.2%           16.1%          8.1%        11.5%          10.5%          26.6%
FN: Missed Defaulter (Danger!)       14.2%           18.4%          7.0%        15.0%          11.7%          33.7%
TN: Clean Approval                   21.9%           17.2%          8.0%        13.8%          10.4%          28.7%
```

### The Human Story of Model Mistakes:
* **Why the model caught defaulters (TP):** `credit_score` dominated with **35.8% of the total decision weight**—severe bureau delinquencies triggered an immediate decline.
* **Why the model missed defaulters (FN):** For borrowers who slipped through, `credit_score` impact collapsed to only **14.2%**. Because their credit bureau report appeared deceptively clean, the model's primary alert system was neutralized, allowing them to be approved despite dangerous debt burdens.

---

## 💻 Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Core Libraries:** `pandas`, `numpy`, `scipy`
* **Machine Learning:** `scikit-learn`, `lightgbm`, `xgboost`
* **Explainability:** `shap`
* **Visualization:** `matplotlib`, `seaborn`

---

## 🚀 How to Run the Notebook

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/credit-risk-default-prediction.git
   cd credit-risk-default-prediction
   ```

2. **Install required dependencies:**
   ```bash
   pip install pandas numpy scikit-learn lightgbm xgboost shap matplotlib seaborn
   ```

3. **Open and run the notebook:**
   Open `default_predictor.ipynb` in VS Code or JupyterLab and run all cells sequentially.

---

## 🔮 Future Enhancements
* **Scorecard Scaling:** Converting LightGBM log-odds into standard FICO-scale points (300–850) using Points to Double Odds (PDO).
* **LGD (Loss Given Default) Modeling:** Building a fractional regression model to estimate recovery percentages on defaulted loans.
* **Expected Loss Engine:** Calculating $EL = PD \times LGD \times EAD$ for risk-based loan pricing.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).

