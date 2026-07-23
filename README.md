# MLOps Predictive Maintenance Classification Pipeline

![Status](https://img.shields.io/badge/status-production--ready-brightgreen) ![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Overview

An end-to-end MLOps pipeline for predictive maintenance classification that demonstrates production-ready machine learning best practices. This project predicts machine failures and failure types using sensor data from heavy equipment, with emphasis on data validation, experiment tracking, drift monitoring, and model explainability.

**Key Achievement:** Achieved **0.7672 macro F1-score** through systematic model selection, hyperparameter optimization, and rigorous evaluation on imbalanced data.

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Architecture](#solution-architecture)
- [Key Features](#key-features)
- [Results & Findings](#results--findings)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Pipeline Stages](#pipeline-stages)
- [Getting Started](#getting-started)
- [Key Learnings](#key-learnings)
- [Contributing](#contributing)

---

## 🏭 Problem Statement

**Business Context:**  
A heavy-equipment manufacturer operates 10,000+ machines on the shop floor. Unplanned downtime costs ₹8-15 lakh per hour, leading to:
- Production delays
- Increased maintenance costs
- Operational inefficiencies

**Challenge:**  
Build a system that predicts whether a machine is operating normally or is likely to fail, and if a failure is expected, identify the specific failure type.

**Data Provided:**
- `train.csv` - Historical labelled training data (10,000 records, highly imbalanced)
- `current.csv` - Stable post-deployment batch (production baseline)
- `stress.csv` - Heavy-load batch (simulating potential data drift)

**Target Classes:**
- No Failure (96.7%) - Normal operation
- Tool Wear Failure (TWF) - 0.43%
- Heat Dissipation Failure (HDF) - 0.98%
- Power-related Failure (PWF) - 0.88%
- Overstress Failure (OSF) - 1.55%

---

## 🏗️ Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MLOps Pipeline: Full Workflow                    │
└─────────────────────────────────────────────────────────────────────┘

1. DATA VALIDATION & EDA
   ├── Pandera Schema Validation
   ├── Class Imbalance Analysis
   ├── Feature Distribution Analysis
   └── Feature Engineering (Power_W, Temp_diff)

2. EXPERIMENT TRACKING & MODEL SELECTION
   ├── Stratified Train-Validation Split (80/20)
   ├── SMOTE Balancing (training split only)
   ├── 4 Baseline Models:
   │   ├── Logistic Regression
   │   ├── Random Forest
   │   ├── XGBoost ✅ Winner
   │   └── LightGBM
   ├── MLflow Experiment Tracking
   └── Macro F1-Score Comparison

3. HYPERPARAMETER TUNING
   ├── Optuna Optimization (30 trials)
   ├── XGBoost Tuning
   ├── Objective: Maximize Macro F1
   └── MLflow Model Registry

4. DRIFT DETECTION & MONITORING
   ├── Current Batch Analysis (Evidently)
   ├── Stress Batch Analysis (Evidently)
   ├── Per-Feature Drift Scores
   └── Retraining Decision Logic

5. MODEL EXPLAINABILITY
   ├── SHAP TreeExplainer
   ├── Per-Class Feature Importance
   ├── Feature Driver Analysis
   └── Physical Interpretation

6. DECISION & RECOMMENDATIONS
   ├── Evidence-Based Conclusions
   ├── Actionable Maintenance Triggers
   └── Monitoring Strategy
```

---

## ✨ Key Features

### 1. **Data Validation with Pandera**
- Schema enforcement for raw sensor inputs
- Domain constraint validation (temperature ranges, speed limits)
- Distinction between schema validity and distribution stability

### 2. **Robust Model Comparison**
- Fair evaluation using stratified splits
- Proper SMOTE application (training split only, k_neighbors=3)
- Macro F1-score as primary metric (handles class imbalance)
- Per-class performance tracking

### 3. **Hyperparameter Optimization**
- Optuna Bayesian optimization (30 trials)
- XGBoost tuning for macro F1 maximization
- 2.5% improvement over baseline

### 4. **Production Monitoring**
- Evidently drift detection
- Feature-level distribution analysis
- Wasserstein distance scoring
- Automated retraining recommendations

### 5. **Model Explainability**
- SHAP TreeExplainer for multiclass models
- Per-failure-class feature importance
- Engineering-level insights
- Physical interpretation of drivers

### 6. **MLflow Integration**
- Complete experiment tracking
- Model versioning and registry
- Production alias management
- Reproducible runs

---

## 📊 Results & Findings

### Stage 1: Data Validation & EDA

**Schema Validation Status:** ✅ All datasets passed validation

**Class Distribution:**
| Failure Type | Count | Percentage |
|-------------|-------|-----------|
| No Failure | 9,688 | 96.88% |
| OSF | 155 | 1.55% |
| HDF | 98 | 0.98% |
| PWF | 88 | 0.88% |
| TWF | 43 | 0.43% |

**Key Insight:** Valid data ≠ Stable data. Pandera validation confirms schema correctness, but Evidently detects distribution shifts (stress.csv).

### Stage 2: Model Selection & Tuning

**Baseline Model Comparison:**

| Model | Macro F1 | Accuracy | Status |
|-------|----------|----------|--------|
| **XGBoost** | **0.7481** | 0.9753 | ✅ Winner |
| Random Forest | 0.7355 | 0.9715 | |
| LightGBM | 0.7296 | 0.9689 | |
| Logistic Regression | 0.5312 | 0.9689 | |

**After Optuna Tuning (XGBoost):**
- **Macro F1:** 0.7672 (+2.5%)
- **Weighted F1:** 0.9828
- **Accuracy:** 0.9771
- **Trials:** 30 Bayesian optimization iterations

**Tuned Hyperparameters:**
```python
{
    'n_estimators': 407,
    'max_depth': 6,
    'learning_rate': 0.0325,
    'min_child_weight': 4.94,
    'subsample': 0.8364,
    'colsample_bytree': 0.9147,
    'gamma': 1.6298,
    'reg_alpha': 3.60e-05,
    'reg_lambda': 0.0092
}
```

**Why Macro F1 Matters:**
- Accuracy (97.7%) is misleading due to extreme class imbalance
- Macro F1 (0.7672) gives equal weight to all failure classes
- In operations: missing rare failures → unplanned downtime

### Stage 3: Drift Detection & Monitoring

**Current Batch (Stable Production):**
- ✅ No drift detected
- ✅ 0 out of 5 features drifted
- **Status:** Stable relative to training baseline

**Stress Batch (Heavy-Load Scenario):**
- ⚠️ Dataset drift detected
- ⚠️ 3 out of 5 features drifted
- **Drifted Features:** Air temperature, Process temperature, Torque
- **Status:** Significant distribution shift

**Interpretation:**
- Stress conditions represent an operationally different regime
- Model staleness risk increases under heavy-load scenarios
- Retraining recommended when stress-like conditions become frequent

### Stage 4: Explainability with SHAP

**Per-Class Feature Importance (Top Drivers):**

| Failure Type | Top Driver | Mechanism |
|-------------|-----------|-----------|
| **TWF** | Tool wear | Direct tool degradation indicator; increases predictably over time |
| **HDF** | Temp_diff | Thermal imbalance signal; indicates cooling system stress |
| **PWF** | Power_W | Combined torque-speed interaction; captures mechanical power loading |
| **OSF** | Torque | Direct load/stress intensity; indicates mechanical overstress |

**Key Finding:** Each failure type has a distinct underlying physical mechanism. Derived features (`Power_W`, `Temp_diff`) outperform raw sensor data.

### Stage 5: Conclusions & Recommendations

**1. Model Selection**
- **Winner:** Tuned XGBoost (macro F1: 0.7672)
- **Justification:** Best balance across all failure classes despite extreme imbalance

**2. Why Accuracy is Misleading**
- Dataset is 96.7% No Failure class
- High accuracy possible by predicting majority class only
- Macro F1 mandatory for imbalanced classification

**3. The TWF Challenge (Data Scarcity)**
- Only 30 real TWF samples in training data
- SMOTE + Optuna cannot fully compensate for lack of real examples
- **Fix:** Collect more labelled TWF failure examples under diverse conditions

**4. Drift Implications**
- Stress batch shows significant drift (3/5 features)
- Model reliability decreases under heavy-load conditions
- Preemptive retraining reduces prediction degradation

**5. Actionable Recommendation**

**Trigger Condition:**
- Sustained increase in `Temp_diff` OR
- Repeated high `Torque` / high `Power_W` operation OR
- Drift detection in incoming batches

**Risked Failure Types:**
- `HDF` under elevated thermal imbalance
- `PWF` under high power consumption
- `OSF` under sustained high torque

**Preventive Action:**
Implement monitoring thresholds and trigger inspection of:
- Cooling systems (thermal management)
- Power transmission components (mechanical load)
- Tool/load conditions (stress resistance)
- Retraining pipelines (when drift detected)

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/barun-khan/mlops-predictive-maintenance.git
cd mlops-predictive-maintenance
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Dependencies

```
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
imbalanced-learn==0.12.4
lightgbm==4.5.0
xgboost==2.1.3
mlflow==2.19.0
optuna==4.1.0
evidently==0.7.21
shap==0.47.0
pandera==0.23.1
matplotlib==3.9.4
seaborn==0.13.2
joblib==1.4.2
```

---

## 📁 Project Structure

```
mlops-predictive-maintenance/
├── MLOps_Assignment_Barun_Khan.ipynb   # Complete pipeline notebook
├── data/
│   ├── train.csv                       # Historical training data
│   ├── current.csv                     # Stable production batch
│   └── stress.csv                      # Heavy-load drifted batch
├── notebooks/
│   └── analysis/                       # Exploratory notebooks
├── artifacts/
│   ├── best_model.pkl                  # Tuned XGBoost model
│   ├── label_encoder.pkl               # Target encoder
│   ├── mlflow.db                       # Experiment tracking DB
│   ├── eda_distributions.png           # Feature distributions
│   ├── drift_current.html              # Evidently report (current)
│   ├── drift_stress.html               # Evidently report (stress)
│   └── shap_per_class.png              # SHAP explainability plot
├── requirements.txt                    # Pinned dependencies
└── README.md                           # This file
```

---

## 🔄 Pipeline Stages

### Stage 1: Data Loading, Schema Validation & EDA

**Objectives:**
- Validate data quality and schema constraints
- Understand class distribution and imbalance
- Engineer meaningful features

**Key Tasks:**
```python
# 1. Load datasets
train = pd.read_csv('data/train.csv')
current = pd.read_csv('data/current.csv')
stress = pd.read_csv('data/stress.csv')

# 2. Define Pandera schema
schema = pa.DataFrameSchema({
    'Air temperature': pa.Column(float, pa.Check.in_range(295.0, 305.0)),
    'Process temperature': pa.Column(float, pa.Check.in_range(305.0, 315.0)),
    'Rotational speed': pa.Column(int, pa.Check.in_range(1000, 2900)),
    'Torque': pa.Column(float, pa.Check.in_range(3.0, 80.0)),
    'Type': pa.Column(str, pa.Check.isin(['L', 'M', 'H'])),
})

# 3. Validate all datasets
schema.validate(train)
schema.validate(current)
schema.validate(stress, lazy=True)

# 4. Feature engineering
train['Power_W'] = train['Torque'] * train['Rotational speed'] * 2 * np.pi / 60
train['Temp_diff'] = train['Process temperature'] - train['Air temperature']
```

**Outputs:**
- `eda_distributions.png` - Feature distributions and class imbalance visualization

### Stage 2: Experiment Tracking & Model Selection

**Objectives:**
- Train and compare multiple models fairly
- Select best model using macro F1
- Track all experiments in MLflow

**Key Tasks:**
```python
# 1. Stratified train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Apply SMOTE (training split only)
smote = SMOTE(k_neighbors=3, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# 3. Train 4 models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'XGBoost': XGBClassifier(n_estimators=100),
    'LightGBM': LGBMClassifier(n_estimators=100)
}

# 4. Log to MLflow
mlflow.set_experiment('PredMaint_ModelSelection')
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train_balanced, y_train_balanced)
        y_pred = model.predict(X_val)
        macro_f1 = f1_score(y_val, y_pred, average='macro')
        mlflow.log_metric('macro_f1', macro_f1)
```

**Outputs:**
- MLflow experiment runs with metrics and model artifacts

### Stage 3: Hyperparameter Tuning with Optuna

**Objectives:**
- Optimize XGBoost for macro F1
- Register best model in MLflow

**Key Tasks:**
```python
# 1. Define Optuna study
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        # ... more hyperparameters
    }
    model = XGBClassifier(**params, random_state=42)
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average='macro')

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=30)

# 2. Register best model
best_model = XGBClassifier(**study.best_params, random_state=42)
best_model.fit(X_train_balanced, y_train_balanced)
mlflow.xgboost.log_model(best_model, 'model')
client = mlflow.tracking.MlflowClient()
client.create_model_version(
    'PredMaint_XGBoost', 
    'runs/{}/model'.format(run_id),
    alias='production'
)
```

**Outputs:**
- `best_model.pkl` - Tuned XGBoost classifier
- `label_encoder.pkl` - Target variable encoder

### Stage 4: Drift Detection & Monitoring

**Objectives:**
- Monitor current and stress batches for distribution shifts
- Make retraining recommendations

**Key Tasks:**
```python
# 1. Evidently drift detection (current batch)
current_report = Report(metrics=[
    DataDriftPreset(),
])
current_report.run(
    reference_data=train[FEATURES],
    current_data=current[FEATURES]
)
current_report.save_html('drift_current.html')

# 2. Evidently drift detection (stress batch)
stress_report = Report(metrics=[
    DataDriftPreset(),
    ColumnDriftMetric(),
])
stress_report.run(
    reference_data=train[FEATURES],
    current_data=stress[FEATURES]
)
stress_report.save_html('drift_stress.html')

# 3. Analyze per-feature drift
for metric in stress_report.as_json()['metrics']:
    if 'column_name' in metric:
        print(f"{metric['column_name']}: Drift={metric['result']['drift_detected']}")
```

**Outputs:**
- `drift_current.html` - Evidently report (current batch)
- `drift_stress.html` - Evidently report (stress batch)

### Stage 5: Explainability with SHAP

**Objectives:**
- Explain model predictions per failure class
- Identify key feature drivers

**Key Tasks:**
```python
# 1. Load best model and compute SHAP values
best_model = joblib.load('best_model.pkl')
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train[FEATURES])

# 2. Create per-class plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
failure_types = ['TWF', 'HDF', 'PWF', 'OSF']

for idx, class_idx in enumerate([0, 1, 2, 3]):
    ax = axes[idx // 2, idx % 2]
    shap.summary_plot(
        shap_values[class_idx], 
        X_train[FEATURES], 
        plot_type='bar',
        show=False,
        ax=ax
    )
    ax.set_title(f'{failure_types[idx]} Feature Importance')

plt.tight_layout()
plt.savefig('shap_per_class.png', dpi=300, bbox_inches='tight')
```

**Outputs:**
- `shap_per_class.png` - 4-panel SHAP feature importance plot

---

## 🎓 Getting Started

### 1. Run the Complete Pipeline

```bash
# Execute the notebook
jupyter notebook MLOps_Assignment_Barun_Khan.ipynb
```

The notebook runs all 5 stages sequentially:
1. Data validation & EDA
2. Model selection & tracking
3. Hyperparameter tuning
4. Drift detection
5. Explainability & conclusions

### 2. View Experiment Results

```python
import mlflow

# Check experiment runs
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=['PredMaint_ModelSelection'])
for run in runs:
    print(f"{run.data.params['model']}: {run.data.metrics['macro_f1']:.4f}")
```

### 3. Review Drift Reports

Open the HTML reports in a browser:
```bash
open drift_current.html
open drift_stress.html
```

### 4. Analyze Model Explanations

The SHAP plot (`shap_per_class.png`) shows:
- Top feature driver per failure class
- Feature value distribution
- Impact magnitude

---

## 💡 Key Learnings

### 1. **Valid Data ≠ Stable Data**
- Pandera validates schema correctness
- Evidently detects distribution shifts
- Must monitor BOTH for production reliability

### 2. **Macro F1 > Accuracy for Imbalanced Problems**
- Accuracy can be deceivingly high (97.7% is misleading)
- Macro F1 gives equal weight to all classes
- Essential when rare failures are operationally critical

### 3. **SMOTE Timing is Critical**
- Apply AFTER train-test split (prevent data leakage)
- Use k_neighbors=3 for tiny rare classes
- Never include validation set in SMOTE

### 4. **Multiclass SHAP Requires Per-Class Analysis**
- Don't collapse classes into single global ranking
- Each failure has distinct feature drivers
- Engineering insights require class-specific interpretation

### 5. **Drift Triggers Retraining Decision**
- Evidence-based (not accuracy-based)
- Connected to business impact
- Monitoring embedded in production workflow

### 6. **Data Scarcity is the Real Challenge**
- 30 real TWF samples cannot be fully solved by tuning
- Synthetic data (SMOTE) helps but doesn't replace real examples
- Operational fix: collect more labelled failure data

---

## 📈 Metrics Summary

| Metric | Value | Status |
|--------|-------|--------|
| **Best Model** | XGBoost (Tuned) | ✅ |
| **Baseline Macro F1** | 0.7481 | ✅ |
| **Tuned Macro F1** | 0.7672 | ✅ |
| **Improvement** | +2.5% | ✅ |
| **Tuned Accuracy** | 0.9771 | ✅ |
| **Current Batch Drift** | None (0/5 features) | ✅ Stable |
| **Stress Batch Drift** | Yes (3/5 features) | ⚠️ Action Required |
| **Models Tracked (MLflow)** | 4 baseline + 1 tuned | ✅ |
| **Optuna Trials** | 30 | ✅ |
| **SHAP Classes Analyzed** | 4 failure types | ✅ |

---

## 🔧 Tools & Technologies

| Category | Tools |
|----------|-------|
| **Data Validation** | Pandera |
| **Experiment Tracking** | MLflow |
| **Hyperparameter Optimization** | Optuna |
| **Drift Detection** | Evidently |
| **Model Explainability** | SHAP |
| **Imbalance Handling** | SMOTE |
| **ML Frameworks** | XGBoost, LightGBM, Random Forest, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |

---

## 📊 Pipeline Workflow

The MLOps loop implemented in this project:

```
validate → train → track → tune → monitor → explain → decide
   ↑                                                      ↓
   └──────────────────── retraining trigger ────────────┘
```

**Stage Flow:**
1. **Validate** - Pandera schema checks
2. **Train** - 4 baseline models with stratified split + SMOTE
3. **Track** - MLflow experiment logging
4. **Tune** - Optuna hyperparameter optimization
5. **Monitor** - Evidently drift detection
6. **Explain** - SHAP per-class feature importance
7. **Decide** - Evidence-based retraining recommendations

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Barun Khan**  
MLOps Engineer | Machine Learning | Production Systems  
[GitHub](https://github.com/barun-khan) | [LinkedIn](https://linkedin.com/in/barun-khan)

---

## 📞 Support

For questions or issues:
- Open a GitHub Issue
- Review the notebook for implementation details
- Check the Evidently HTML reports for drift analysis

---

## 📚 References

- [Pandera Documentation](https://pandera.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Evidently Documentation](https://docs.evidentlyai.com/)
- [SHAP Documentation](https://shap.readthedocs.io/)

---

**Last Updated:** July 24, 2026  
**Status:** ✅ Production Ready | ✅ Passed Capstone Exam

---

## 🎯 Quick Links

- [📓 Notebook](./MLOps_Assignment_Barun_Khan.ipynb) - Complete implementation
- [📊 Data](./data/) - Training and production batches
- [🔍 Drift Reports](./artifacts/drift_*.html) - Evidently monitoring
- [🎨 SHAP Plot](./artifacts/shap_per_class.png) - Model explainability
- [🤖 Best Model](./artifacts/best_model.pkl) - Production model

