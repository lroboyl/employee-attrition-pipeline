# Data Drift Detection and Monitoring

This document summarizes the results of drift detection for the HR Attrition model and provides recommendations for maintaining model performance over time.

## Setup

The drift monitoring script (`src/monitor_drift.py`) compares the original training data (reference dataset) against a simulated production dataset using Evidently.

**Reference dataset:** 70% of the IBM HR Analytics dataset created using `train_test_split` (the same split used for model training).

**Production dataset:** The remaining 30% of records (~441 employees) with synthetic drift injected into six important features.

| Feature            | Drift Applied                                                 |
| ------------------ | ------------------------------------------------------------- |
| `Age`              | Shifted -5 years (younger workforce)                          |
| `MonthlyIncome`    | Increased 10–25% (inflation and market-rate adjustments)      |
| `DistanceFromHome` | Increased exponentially (relocation or hybrid-work rollbacks) |
| `YearsAtCompany`   | Reduced by ~2 years (higher turnover)                         |
| `WorkLifeBalance`  | Shifted toward lower ratings (more burnout)                   |
| `OverTime`         | 20% more employees changed to "Yes"                           |

---

## Q1: Which features showed drift and why?

The drift monitoring script detected **15 out of 30 features drifted (50% overall drift share)**.

The following features were intentionally modified and showed statistically significant drift.

### High-Drift Features (p-value < 0.0001)

#### `Age`

A five-year decrease in average age creates a noticeable shift in the distribution. This simulates a workforce becoming younger over time. Since age is related to career stage and tenure expectations, changes in this feature can influence attrition patterns.

#### `MonthlyIncome`

Increasing salaries by 10–25% creates a substantial shift in the income distribution. Compensation is one of the strongest predictors of attrition, so changes in salary levels can directly affect model predictions.

#### `DistanceFromHome`

Adding exponential growth increases the number of employees with longer commutes. This reflects real-world scenarios such as relocation trends or hybrid-work rollbacks that require employees to travel farther.

#### `YearsAtCompany`

Reducing tenure by approximately two years simulates a labor market with increased employee turnover. Since tenure is a highly predictive feature, drift in this variable is especially important to monitor.

#### `WorkLifeBalance`

The distribution is shifted toward lower ratings (1 and 2), representing increased employee burnout. The shift was strong enough to produce a near-zero p-value.

#### `OverTime`

Increasing the proportion of employees working overtime changes the distribution of this binary feature significantly. Because `OverTime` is a strong attrition predictor, even moderate drift may influence model behavior.

### Unexpected Features That Also Drifted

Nine additional features showed statistically significant drift that was not directly injected — including `MaritalStatus`, `NumCompaniesWorked`, `JobRole`, `Education`, `EducationField`, `StockOptionLevel`, `JobInvolvement`, `TrainingTimesLastYear`, and `PercentSalaryHike`. These likely reflect correlations with the injected features (for example, income correlates with job level and education) or natural sampling variation in the 30% holdout split.

---

## Q2: Would this drift affect model performance?

**Yes, likely.**

Several of the drifted features (`Age`, `MonthlyIncome`, `YearsAtCompany`, and `OverTime`) are among the most important predictors in the Random Forest model.

The model was trained on data where:

* Employee ages were concentrated around 35–40 years old
* Monthly income commonly ranged between $4,000–6,000
* Roughly 30% of employees worked overtime

The simulated production data no longer follows those same patterns.

Because Random Forest models learn decision boundaries from the training distribution, significant distributional shifts can lead to less reliable predictions and poorer probability calibration.

Potential impacts include:

* **Under-predicting attrition** for younger employees with shorter tenure.
* **Weakening income-related signals** as salary distributions increase.
* **Increasing false positives** if overtime becomes more common across the workforce.

While the exact impact would require validation on real production data, a noticeable reduction in model performance metrics such as ROC-AUC would be expected if the model were deployed without retraining.

---

## Q3: What action should be taken?

### Recommendation: Retrain on Recent Data and Continue Monitoring

| Scenario                                               | Recommended Action                               |
| ------------------------------------------------------ | ------------------------------------------------ |
| `MonthlyIncome`, `Age`, or `YearsAtCompany` show drift | **Retrain** – these are high-importance features |
| Only lower-impact features drift                       | **Monitor** and continue collecting data         |
| Overall drift share > 30%                              | **Retrain immediately**                          |
| Overall drift share 15–30%                             | **Investigate root causes**                      |
| Overall drift share < 15%                              | **Continue monitoring**                          |

### Next Steps

1. **Retrain the model** using the most recent HR records to capture current workforce demographics and compensation trends.

2. **Review the classification threshold.** The default threshold of 0.5 may no longer provide the best balance between precision and recall after the distribution shift.

3. **Schedule regular drift monitoring.** Run `src/monitor_drift.py` weekly and generate alerts whenever drift exceeds the configured threshold.

4. **Investigate business drivers behind the drift.** For example, if `DistanceFromHome` changes because of workplace policy updates, the shift may be permanent and require new features or model retraining.

---

## Running the Drift Monitor

```bash
# From project root
pip install -r requirements.txt
python src/monitor_drift.py configs/config.yaml
```

The script will:

* Split the dataset into disjoint reference and production datasets using `train_test_split`
* Inject synthetic drift into six selected features
* Run Evidently drift detection across all features
* Log results to stdout using Python's `logging` module
* Save an HTML report to `reports/drift_report_<timestamp>.html`
* Retain the 10 most recent reports
* Exit with code 1 if drift exceeds the configured threshold (default: 30%)

After the script finishes, open the generated HTML report to explore detailed drift metrics, distribution comparisons, and the overall drift score.

```bash
open reports/drift_report_<timestamp>.html        # macOS
start reports/drift_report_<timestamp>.html       # Windows
xdg-open reports/drift_report_<timestamp>.html    # Linux
```

The HTML report provides a visual breakdown of feature drift, making it easy to identify which variables changed the most and whether retraining should be considered.
