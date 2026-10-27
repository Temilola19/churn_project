# Telco Customer Churn: Who Leaves, Why, and What to Do About It

An end-to-end analysis of 7,043 telecom customers: data audit → EDA → statistical testing → predictive model → a profit-based retention strategy.

**Dataset:** IBM Telco Customer Churn (public sample, 7,043 customers × 21 columns). Downloaded from `raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d`.

## The business question
Churn is 26.5% of customers and about 30% of monthly recurring revenue (~$139K of ~$456K). **Where is it concentrated, can we predict it, and is a targeted retention campaign worth running?**

## Key findings

1. **Churn is concentrated, not diffuse.** Month-to-month customers on fiber are 30% of the base but churn at **54.6%** and account for **72% of all lost monthly revenue**.
2. **Contract type is the strongest signal.** Month-to-month customers churn at 6.3× the rate of contracted customers (95% CI 5.5–7.2). Cramér's V = 0.41, the highest of any feature.
3. **Churn is an early-life problem.** Median tenure is 10 months for churners vs 38 for stayers (Mann-Whitney p < 1e-200). Customers in their first 6 months churn at very high rates.
4. **The fiber effect is not a tenure artifact.** Fiber churns more than DSL in *every* tenure bucket (e.g. 74% vs 48% at 0–6 months, 37% vs 11% at 25–48 months), so newer fiber customers aren't simply driving it.
5. **Things that don't matter:** gender and phone service show no significant association with churn after Holm-Bonferroni correction.
6. **Mitigating factors:** online security, auto-pay, and longer contracts are associated with lower churn. Electronic-check payers and paperless billing are associated with higher churn.

## Model results (held-out test set, n = 1,409)

| Model | CV AUC (train, 5-fold) | Test AUC | Test PR-AUC | Brier |
|---|---|---|---|---|
| Baseline (prior) | 0.500 | 0.500 | 0.265 | 0.195 |
| Logistic regression | 0.845 ± 0.012 | 0.842 | 0.635 | 0.138 |
| **Gradient boosting** | **0.846 ± 0.013** | **0.844** (95% CI 0.823–0.865) | **0.661** | **0.136** |

- The two models are statistically indistinguishable on AUC. **Logistic regression is the pragmatic production choice** (interpretable, odds ratios in `figures/10`); boosting is marginally better on PR-AUC and calibration.
- Top-decile lift is **2.9×**; the top 20% of customers by risk contain **49%** of all churners.
- Drivers (logistic odds, per 1 SD for numerics): fiber optic, month-to-month, paperless billing, and electronic check raise risk; tenure, no internet service, two-year contract, and phone service lower it.

## Business value: choosing who to contact
Assumptions (editable in `src/config.py`, **hypothetical, not from the data**): $100 cost per retention offer, 25% of true churners are saved, saved customers pay 12 more months.

| Strategy | Test-set profit |
|---|---|
| Do nothing | $0 |
| Contact everyone | **−$59,255** |
| Model-targeted (P(churn) ≥ 0.48, contacts 23% of customers) | **+$15,661** |

Threshold was chosen on out-of-fold *training* predictions, then scored once on test. Profit is flat between thresholds of roughly 0.40 and 0.60, so the decision is not fragile to the exact cut-off. At the chosen point precision is 65% and recall 55%.

## Recommendations
1. **Prioritize month-to-month fiber customers**, the single segment holding most revenue at risk, for contract-conversion offers (e.g. a discount in exchange for a 1-year term).
2. **Invest in the first 6 months**: onboarding checks and proactive outreach for new fiber customers.
3. **Nudge payment method**: test incentives to move electronic-check customers to auto-pay.
4. **Bundle security/tech support** for at-risk customers; test it rather than assume it works (see caveats).
5. **Run an A/B test before scaling.** The 25% save rate is an assumption; the experiment measures the real number and tells us whether the profit estimate holds.

## Limitations (read before acting)
- **Observational data, one snapshot.** Associations ≠ causation. Add-on services may simply attract more committed customers. Fiber's churn may reflect price, service quality, or competitors, which this data can't separate.
- **No time dimension or churn reason.** Can't model survival curves properly, seasonality, or drift. Leakage risk is limited (no post-churn fields) but `TotalCharges` is tenure-dependent.
- **Profit numbers depend entirely on assumed campaign economics.** Treat them as a framework, not a forecast.
- Single train/test split on ~7K rows; the AUC confidence interval is ±0.02.
- The dataset is a well-known public sample, and a clean one. Real data needs far more cleaning and validation.

## Data quality notes
No duplicates. `TotalCharges` is stored as text because 11 blank values exist, all with tenure = 0 (new customers with no billing yet) → set to 0, not dropped. Redundant "No internet/phone service" categories collapsed to "No". See `reports/data_quality_log.json`.

## Project structure
```
data/            raw + cleaned CSV
src/
  config.py      paths, seed, business assumptions
  01_clean.py    audit, type fixes, feature engineering
  02_eda.py      segment plots, chi-square/Cramér's V (Holm-corrected), Mann-Whitney, risk ratio, revenue at risk
  03_model.py    CV, test eval, bootstrap CI, explainability, profit-based threshold, lift
reports/         JSON/CSV results + figures/ (13 charts)
run_all.py       reproduces everything (~30s)
```
Run: `pip install -r requirements.txt && python run_all.py`

## Methods choices worth noting
- Test set is touched once; model and threshold are selected using training data only.
- Multiple-testing correction (Holm) over 17 categorical tests; effect sizes reported alongside p-values because with n=7K almost everything is "significant".
- Stratified splits; PR-AUC reported because of class imbalance (26.5% positive).
- Business-metric evaluation, not just accuracy.
