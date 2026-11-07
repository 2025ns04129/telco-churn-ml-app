# Telco Customer Churn — Five-Model Comparison

**DSE ZG565 Machine Learning — Assignment 2**

A comparison of the five classifiers from the course syllabus — Logistic Regression,
Decision Tree, k-Nearest Neighbour, Gaussian Naive Bayes and a Support Vector Machine —
on the **Telco Customer Churn** dataset, served through an interactive Streamlit app.

## Summary

All five classifiers were trained and tuned (5-fold cross-validation) to predict customer
churn on an imbalanced dataset (26.5% churn), where plain accuracy is misleading because a
"predict everyone stays" model already scores 73.5%. **No single model dominates every
metric:** the **SVM (RBF)** gives the best balanced performance (F1 0.6216, MCC 0.4659) and
catches 285 of 371 churners, **Logistic Regression** ranks churn risk best (AUC 0.8467),
**kNN** raises the fewest false alarms (precision 0.6228) but misses the most churners, and
**Naive Bayes** catches almost every churner (recall 0.8356) at the cost of many false alarms.
The choice is therefore a **recall-versus-precision trade-off**; because missing a churner
usually costs more than a wasted retention offer, **SVM is recommended as the headline model**,
with Logistic Regression as a lightweight, interpretable alternative.

---

## a. Problem statement

A telecom company loses revenue every time a customer **churns** (closes their account).
Retaining an existing customer is far cheaper than acquiring a new one, so the company wants
to know *in advance* which customers are likely to leave, in order to target them with
retention offers.

This is a **binary classification** problem: given a customer's account, service and billing
profile, predict whether they will churn (`Churn = Yes`) or stay (`Churn = No`). The five
syllabus classifiers are trained, tuned and compared on six evaluation metrics to see which
best supports a retention campaign.

## b. Dataset description

**Source:** Telco Customer Churn — a widely used customer-retention dataset published by IBM
and mirrored on Kaggle.
Download used in `model/prepare_data.py`:
`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`

| Property | Value |
|---|---|
| Instances (after cleaning) | **7,010** customers |
| Features | **19** (3 numeric + 16 categorical) |
| Target | `Churn` (Yes / No) |
| Class balance | 5,153 No (73.5%) / 1,857 Yes (26.5%) — **imbalanced** |
| Train / test split | 5,608 / 1,402 (80 / 20, stratified, `random_state=42`) |

Both the instance count (≥ 500) and the feature count (≥ 12) satisfy the assignment minimums.

### Features

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | gender | categorical | Male / Female |
| 2 | SeniorCitizen | categorical | Whether the customer is a senior citizen |
| 3 | Partner | categorical | Has a partner |
| 4 | Dependents | categorical | Has dependents |
| 5 | tenure | numeric | Months the customer has stayed with the company |
| 6 | PhoneService | categorical | Has phone service |
| 7 | MultipleLines | categorical | Multiple phone lines / no phone service |
| 8 | InternetService | categorical | DSL / Fiber optic / No |
| 9 | OnlineSecurity | categorical | Has online security add-on |
| 10 | OnlineBackup | categorical | Has online backup add-on |
| 11 | DeviceProtection | categorical | Has device protection add-on |
| 12 | TechSupport | categorical | Has tech-support add-on |
| 13 | StreamingTV | categorical | Has streaming TV |
| 14 | StreamingMovies | categorical | Has streaming movies |
| 15 | Contract | categorical | Month-to-month / One year / Two year |
| 16 | PaperlessBilling | categorical | Uses paperless billing |
| 17 | PaymentMethod | categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| 18 | MonthlyCharges | numeric | Current monthly charge |
| 19 | TotalCharges | numeric | Lifetime charges to date |

### Cleaning (`model/prepare_data.py`)

- Dropped `customerID` — a unique identifier with no predictive value.
- `TotalCharges` arrives as text; **11 brand-new customers** (tenure 0) hold a blank string.
  These were coerced to numeric and dropped.
- `SeniorCitizen` was recoded from `0/1` to `No/Yes` so every categorical feature is
  human-readable in the shipped CSV.
- **22 exact duplicate rows** removed after dropping the ID column.
- Result: 7,043 → **7,010** rows.

### Preprocessing (inside every model pipeline)

A single `ColumnTransformer` **standardises the 3 numeric features** (`StandardScaler`) and
**one-hot encodes the 16 categorical features** (`OneHotEncoder(handle_unknown="ignore")`),
expanding to 46 encoded columns. It is wrapped inside each model's `Pipeline`, so the scaler
and encoder are fitted **only on the training folds** during cross-validation and replayed
identically at prediction time — there is no leakage from the test set, and the app can score
a raw, human-readable customer CSV directly.

## c. GitHub repository link

<https://github.com/2025ns04129/telco-churn-ml-app>

```
telco-churn-ml-app/
├── app.py                        # Streamlit front-end
├── requirements.txt              # pinned dependencies
├── README.md                     # this file
├── test_data.csv                 # 1,402-row stratified hold-out split (used in the app)
├── .python-version               # pins Python 3.12 for Streamlit Cloud
├── data/
│   ├── telco_churn_full.csv      # cleaned full dataset (7,010 × 20)
│   └── train_data.csv            # 5,608-row training split
└── model/
    ├── prepare_data.py           # download + clean + split
    ├── train_models.py           # train, tune and persist the five pipelines
    ├── telco_churn_training.ipynb# notebook version (for the BITS Virtual Lab)
    ├── *.joblib                  # the five trained pipelines
    ├── model_metrics.csv         # the comparison table
    └── model_registry.json       # metadata the app reads
```

## d. Models used

All five pipelines were tuned with **5-fold cross-validation**, scored by **F1 on the churn
class**. `class_weight="balanced"` was offered to the three models that support it, because
the data is imbalanced and recall on churners is what a retention campaign cares about.

| Model | Contact session | Tuned hyper-parameters | CV churn-F1 |
|---|---|---|---|
| Logistic Regression | 4 & 5 | `C=10.0`, `class_weight=balanced` | 0.6316 |
| Decision Tree (entropy) | 6 | `max_depth=5`, `min_samples_leaf=20`, `class_weight=balanced` | 0.6217 |
| k-Nearest Neighbour | 7 | `n_neighbors=31`, `weights=uniform` | 0.6042 |
| Gaussian Naive Bayes | 11 | `var_smoothing=1e-9` | 0.5938 |
| Support Vector Machine (RBF) | 9 | `C=1.0`, `class_weight=balanced` | 0.6247 |

### Comparison table (hold-out test set, n = 1,402)

Metrics are reported on the **churn (`Yes`) class**; AUC is the ROC AUC on the churn
probability. **Best in each column is bold.**

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7425 | **0.8467** | 0.5088 | 0.7763 | 0.6147 | 0.4555 |
| Decision Tree | 0.7582 | 0.8312 | 0.5309 | 0.7412 | 0.6187 | 0.4620 |
| kNN | **0.7939** | 0.8361 | **0.6228** | 0.5606 | 0.5901 | 0.4540 |
| Naive Bayes | 0.7019 | 0.8138 | 0.4648 | **0.8356** | 0.5973 | 0.4322 |
| **SVM (RBF)** | 0.7525 | 0.8215 | 0.5220 | 0.7682 | **0.6216** | **0.4659** |

No single model dominates — the right choice depends on the business goal:

- **SVM (RBF)** wins on **F1 and MCC**, the two metrics that best summarise balanced
  performance on an imbalanced problem. On the test set it catches **285 of 371 churners
  (recall 0.768)** while keeping precision at 0.522.
- **kNN** has the **highest accuracy and precision** — it raises the fewest false alarms, but
  misses the most churners (recall 0.561), which is the costly error here.
- **Logistic Regression** has the **best AUC (0.8467)** — the best pure ranking of customers
  by churn risk, useful if you want to target the top-N riskiest.
- **Naive Bayes** has the **highest recall (0.836)** — it flags almost every churner, at the
  cost of the lowest accuracy and precision (many false alarms).

### Why accuracy alone is misleading

The dataset is **26.5% churn**, so a model that blindly predicts "No" for everyone already
scores **73.5% accuracy** while catching *zero* churners. That is why the balanced models
show *lower* accuracy than kNN yet are more useful: they accept more false alarms in exchange
for finding the customers who actually leave. **F1, AUC and MCC** are the honest yardsticks.

## Observations

| # | Observation | Evidence |
|---|---|---|
| 1 | **Contract type is the strongest churn driver.** | Month-to-month customers churn at **42.6%**, one-year at 11.3%, two-year at just **2.8%**. |
| 2 | **Fiber-optic internet customers churn far more** than DSL or no-internet. | Fiber optic **41.8%** vs DSL 18.9% vs no internet 7.2%. |
| 3 | **Electronic-check payers are the highest-risk billing segment.** | Electronic check **45.1%** churn vs ~15–19% for automatic / mailed methods. |
| 4 | **Short tenure predicts churn; long tenure protects against it.** | `tenure` has the strongest numeric correlation with churn (**−0.35**); new customers leave most. |
| 5 | **Lack of add-on services (tech support, online security) raises churn.** | No tech support → **41.5%** churn vs 15.2% for customers who have it. |
| 6 | **The models cluster in a narrow band** (AUC 0.81–0.85), typical for this well-studied dataset; the decision between them is a recall-vs-precision trade-off, not a clear accuracy winner. | See comparison table. |

## How the app meets the Section-4 requirements

| Required feature | Implemented as |
|---|---|
| a. Dataset upload option (CSV) | Sidebar → *1. Test data* → **Upload a test CSV** |
| b. Model selection dropdown | Sidebar → *2. Model* → **Classification model** (all 5) |
| c. Display of evaluation metrics | Six metric cards + all-five-model comparison table and bar chart |
| d. Confusion matrix / classification report | Tabs → **Confusion matrix** (2×2 heat-map) and **Classification report** |

The app also shows a data preview with the churn distribution, per-row churn probabilities
with a downloadable predictions CSV, and the reference training scores for a sanity check.

## Reproducing the results

```bash
pip install -r requirements.txt

# 1. download + clean + split  -> data/*.csv and test_data.csv
python model/prepare_data.py

# 2. train, tune and persist   -> model/*.joblib, model_metrics.csv, model_registry.json
python model/train_models.py

# 3. run the app
streamlit run app.py
```

Everything is seeded with `random_state=42`, so the splits, tuned parameters and metrics
reproduce exactly.

## Deploying on Streamlit Community Cloud

1. Push this repository (public) to GitHub.
2. At <https://streamlit.io/cloud>, **New app** → select the repo → branch `main` →
   main file `app.py`.
3. **Open Advanced settings and set the Python version to 3.12 before deploying.**
   `numpy==2.5.2` requires Python ≥ 3.12, so on an older interpreter the build fails at
   `pip install` — the most common cause of deployment failure. A `.python-version` file is
   included to request 3.12 automatically.

## References

1. Telco Customer Churn dataset — IBM Cognos Analytics sample data / Kaggle
   (`blastchar/telco-customer-churn`).
2. scikit-learn: Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, 2011.
3. DSE ZG565 Machine Learning — course handout and contact-session material.
