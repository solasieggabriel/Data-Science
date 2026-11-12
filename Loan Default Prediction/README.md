# Risk and Default Prediction Framework
### *Benchmarking Classical Baselines, Ensembles, and Neural Networks with SHAP*

---

## Summary

This repository presents **Risk Modeling and Default Prediction Framework** built on a synthetic 50,000-borrower financial portfolio with real-world statistical properties (heavy Pareto tails, zero-inflated delinquency counts, and complex missing mechanisms). 

Addressing the standard banking challenge of class imbalance (~11% default rate), this project demonstrates:
1. **Empirical Missing Data Auditing:** Diagnosing **MCAR** vs. **MAR** mechanisms and designing a leakage-free imputation architecture.
2. **Hybrid Column Transformation:** Selectively pairing non-linear power transformations (`Yeo-Johnson`) for skewed financial amounts with `RobustScaler` and `OneHotEncoder`.
3. **5-Model Cross-Validation Tournament:** Evaluating **Logistic Regression**, **Random Forest**, **Multi-Layer Perceptron (MLP)**, **XGBoost**, and **LightGBM** across 5-fold stratified splits, prioritizing **PR-AUC (Precision-Recall AUC)**.

---

## Project Architecture

```
[ 1. Realistic Synthesis ] ──► [ 2. Missingness Audit ] ──► [ 3. Leakage-Free Preprocessing ]
  • Pareto/Log-normal income     • Credit Score: MCAR        • Stratified Train/Test Split (80/20)
  • Zero-inflated delinquencies  • Employment Length: MAR    • Age-bracket median imputation
  • Imbalanced defaults (~11%)     (50% spike at 18-24)      • Hybrid ColumnTransformer

                                          │
                                          ▼
[ 6. Explainable AI and Audit ] ◄── [ 5. Model Tournament ] ◄── [ 4. Metric Selection ]
  • Global Beeswarm Plot          • 5-Fold Stratified CV       • Prioritizing PR-AUC for
  • Unscaled Waterfall Reports    • Baseline vs. Challengers    minority default detection
  • Confusion Matrix Slicing
          
```

---

## 1. The Dataset and Missing Mechanism Diagnosis

Financial data in production is never normally distributed. The dataset mimics real-world banking characteristics:
* **`annual_income`:** Log-normal distribution with fat Pareto tails (0.5% extreme earners up to \$3,000,000).
* **`dti_ratio`:** Gamma-distributed with severe distress outliers (> 0.8 up to 2.5).
* **`revolving_utilization`:** Beta-distributed with over-limit cardholders (> 100%).

### Missingness Audit (Empirical Proof of MAR vs. MCAR)
Before imputing, audited missing rates across demographic quintiles:
* **`credit_score`:** Consistently flat (~3.5%) across all age, income, and housing brackets $\rightarrow$ **MCAR (Missing Completely at Random)**.
* **`employment_length_years`:** Exhibited a massive spike to **~50% in the 18–24 age cohort** (students and entry-level workers) and flat 5% elsewhere $\rightarrow$ **MAR (Missing at Random) conditioned on Age**.

---

## Preprocessing and Leakage Prevention Pipeline

1. **Strict Partitioning First:** Split into Train (80%) and Test (20%) using stratified sampling before calculating any transformation parameters.
2. **Context-Aware Imputation:**
   * `employment_length_years`: Imputed using the **median of the applicant's specific age cohort** learned exclusively from the training set.
   * `credit_score`: Imputed with the global training median.
   * **Missingness Indicators:** Created binary flags (`credit_score_was_missing`, `emp_len_was_missing`) to preserve the "thin-file" risk.
3. **Hybrid `ColumnTransformer` Architecture:**
   * **`PowerTransformer(method='yeo-johnson')`:** Applied to heavily skewed dollar amounts (`annual_income`, `loan_amount`, `dti_ratio`, `avg_tx_amount_30d`) to normalize variance and linearize log-odds.
   * **`RobustScaler()`:** Applied to bounded continuous features (`credit_score`, `open_credit_lines`, `revolving_utilization`) using median and IQR to neutralize extreme outlier distortion.
   * **`OneHotEncoder(drop='first')`:** Applied to nominal categories (`home_ownership`, `loan_intent`).
   * **`passthrough`:** Preserved binary flags as pure 0/1 inputs.

---

## Model Benchmark and Evaluation Tournament

Because defaults represent only ~11% of applicants, established **PR-AUC (Precision-Recall AUC)** as primary objective metric, ran under **5-Fold Stratified Cross-Validation**:

| Rank | Model Paradigm | CV PR-AUC (Mean) | CV PR-AUC (Std) | CV ROC-AUC (Mean) | Test PR-AUC | Test ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **LightGBM (Leaf-Wise Boosting)** | **0.6315** | $\pm 0.0214$ | **0.8772** | **0.6395** | **0.8787** |
| 2 | **XGBoost (Level-Wise Boosting)** | 0.6301 | $\pm 0.0219$ | 0.8764 | 0.6375 | 0.8791 |
| 3 | **Neural Network (MLP: 64-32)** | 0.6241 | $\pm 0.0192$ | 0.8764 | 0.6318 | 0.8739 |
| 4 | **Logistic Regression (Baseline)** | 0.6060 | $\pm 0.0231$ | 0.8700 | 0.6062 | 0.8695 |
| 5 | **Random Forest (Bagging)** | 0.5905 | $\pm 0.0183$ | 0.8728 | 0.5975 | 0.8731 |

### Key Benchmark Findings:
*  All 5 models scored virtually identical ROC-AUC scores (~0.870 to 0.879), while PR-AUC showed a **4.9 percentage point performance gap** between Random Forest and LightGBM.
*  Every model’s Cross-Validation PR-AUC matched its Test score within $\pm 0.008$, proving that the preprocessing pipeline suffered **zero data leakage**.
*  Sequential residual learning (LightGBM/XGBoost) significantly outperformed parallel tree averaging (Random Forest) in isolating minority default risk.

---

## TreeSHAP and Banking Compliance

### A. Global Risk Drivers (Beeswarm Analysis)
Over 70% of the model's total decision weight is driven by four core fundamentals:
1. `credit_score`
2. `annual_income`
3. `dti_ratio`
4. `revolving_utilization`

### B. Local Adverse Action Notices (Unscaled Waterfall Explanations)
By mapping SHAP values back to original real-world units, the framework automatically produces legally compliant rejection reasons:
* *Applicant Example:* Declined with **72.0% Predicted Default Probability**.
  * Primary Reason: `dti_ratio = 0.85` (+32.5% risk contribution)
  * Secondary Reason: `revolving_utilization = 1.05` (+21.0% risk contribution)
  * Tertiary Reason: `credit_score = 540` (+18.4% risk contribution)

---

## Confusion Matrix Error Slicing (Auditing Model Failures)

By breaking down mean feature impact across the four confusion matrix quadrants, audited the root causes of model accuracy and failure:

```
=== PERCENTAGE SHARE OF TOTAL DECISION IMPACT (%) ===
Quadrant                          credit_score   annual_income   dti_ratio   loan_amount   revolving_util   All Other
TP: Caught Defaulter                 35.8%           15.1%         12.1%        10.4%           8.3%          18.3%
FP: False Alarm (Lost Customer)      27.2%           16.1%          8.1%        11.5%          10.5%          26.6%
FN: Missed Defaulter (Danger!)       14.2%           18.4%          7.0%        15.0%          11.7%          33.7%
TN: Clean Approval                   21.9%           17.2%          8.0%        13.8%          10.4%          28.7%
```
![Top Decision Drivers per Quadrant](Decision%20Drivers%20per%20Quadrant.png)

* **Why the model caught defaulters (TP):** `credit_score` dominated with **35.8% of the total decision weight**—severe bureau delinquencies triggered an immediate decline.
* **Why the model missed defaulters (FN):** For borrowers who slipped through, `credit_score` impact collapsed to only **14.2%**. Because their credit bureau report appeared deceptively clean, the model's primary alert system was neutralized, allowing them to be approved despite dangerous debt burdens.
