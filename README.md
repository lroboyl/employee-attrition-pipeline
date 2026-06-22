# HR Employee Attrition Prediction with MLOps

This project predicts employee attrition using the IBM HR Analytics dataset and demonstrates an end-to-end MLOps workflow for training, tracking, testing, and monitoring machine learning models.

The goal is not only to build an accurate model, but also to show how machine learning systems can be managed in a production-like environment using modern MLOps practices.

## Project Highlights

* Predict employee attrition using machine learning
* Track experiments and model performance with MLflow
* Version datasets with DVC
* Automate testing and training with GitHub Actions
* Monitor production data drift with Evidently
* Enforce model quality thresholds before deployment

## Technologies Used

| Category            | Tools          |
| ------------------- | -------------- |
| Programming         | Python         |
| Machine Learning    | Scikit-learn   |
| Experiment Tracking | MLflow         |
| Data Versioning     | DVC            |
| CI/CD               | GitHub Actions |
| Monitoring          | Evidently      |
| Testing             | Pytest         |

---

## Project Structure

```text
employee-attrition-p/
├── .github/
│   └── workflows/
│       └── pipeline.yml
├── configs/
│   └── config.yaml
├── data/ 
├── models/
├── reports/
├── src/
│   ├── compare_experiments.py
│   ├── drift_monitoring.py
│   ├── evaluate.py
│   ├── preprocess.py
│   └── train.py
├── tests/
│   └── test_suite.py
├── MONITORING.md
├── README.md
└── requirements.txt
```
> **Note:**
> - `data/`, `models/`, and `reports/` are intentionally empty in the repository.
> - `data/` is managed by DVC — run `dvc pull` to download the dataset after cloning.
> - `models/` and `reports/` are populated at runtime by the training and drift monitoring scripts.

---

## Dataset

**IBM HR Analytics Employee Attrition Dataset**

* 1,470 employee records
* 35 features
* Binary classification target (`Attrition`)
* Class imbalance: ~84% No, ~16% Yes

The target variable predicts whether an employee is likely to leave the company.

---

## Model Pipeline

The training pipeline performs the following steps:

1. Load and validate the dataset
2. Drop constant-value columns
3. Fill missing values
4. Encode categorical features
5. Scale numerical features
6. Split data using stratified sampling
7. Train the selected model
8. Evaluate model performance
9. Log results to MLflow
10. Save model artifacts for inference

---

## Quick Start

### Prerequisites

* Python 3.11+
* Git
* DVC

### Clone the Repository

```bash
git clone https://github.com/<your-username>/hr-attrition-mlops.git
cd hr-attrition-mlops
```

### Create a Virtual Environment

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Retrieve the Dataset

```bash
dvc pull
```

The dataset is stored with DVC and is not committed directly to Git.

### Run Tests

```bash
pytest tests/ -v
```

### Train the Model

```bash
python src/train.py configs/config.yaml
```

Training automatically:

* Loads configuration values from `config.yaml`
* Trains the selected model
* Logs parameters and metrics to MLflow
* Saves model artifacts
* Enforces minimum performance thresholds

---

## Experiment Tracking

MLflow is used to track:

* Model parameters
* Evaluation metrics
* Model artifacts
* Experiment history

### Compare Experiments

```bash
python src/compare_experiments.py configs/config.yaml
```

Optional arguments:

```bash
python src/compare_experiments.py configs/config.yaml --metric metrics.f1
python src/compare_experiments.py configs/config.yaml --top 10
```

The script ranks runs and identifies the best-performing model based on the selected metric.

---

## Configuration

All project settings are managed through:

```text
configs/config.yaml
```

This includes:

| Section      | Purpose                               |
| ------------ | ------------------------------------- |
| `data`       | Dataset paths and train/test settings |
| `features`   | Feature definitions                   |
| `model`      | Model type and hyperparameters        |
| `mlflow`     | Tracking configuration                |
| `thresholds` | Minimum performance requirements      |
| `output`     | Saved model and report locations      |

Because configuration is separated from code, experiments can be repeated simply by changing values in `config.yaml`.

---

## CI/CD Pipeline

GitHub Actions automatically runs the machine learning pipeline on every push and pull request.

### Test Stage

* Install dependencies
* Pull dataset using DVC
* Run Pytest test suite

### Training Stage

* Train model
* Evaluate performance
* Fail the pipeline if thresholds are not met

This creates a simple quality gate that prevents low-performing models from progressing through the workflow.

---

## Drift Monitoring

The project includes automated drift detection using Evidently.

```bash
python src/drift_monitoring.py configs/config.yaml
```

The monitoring workflow:

1. Creates reference and production datasets
2. Injects realistic synthetic drift
3. Runs Evidently drift detection
4. Generates an HTML report
5. Fails if drift exceeds the configured threshold

Reports are saved to:

```text
reports/
```

For a detailed analysis of detected drift and recommended actions, see:

```text
MONITORING.md
```

---

## Key MLOps Concepts Demonstrated

This project demonstrates:

* Data versioning with DVC
* Reproducible training pipelines
* Configuration-driven experimentation
* Experiment tracking with MLflow
* Automated testing with Pytest
* CI/CD automation with GitHub Actions
* Model quality gates
* Data drift monitoring with Evidently

---

## Future Improvements

Potential enhancements include:

* Model registry integration
* Automated retraining workflows
* Containerization with Docker
* Cloud deployment
* Real production monitoring dashboards
* Feature store integration

---

## Author

Lauren O'Boyle

Former Physical Education Teacher transitioning into Data Science and Machine Learning, with a focus on building practical machine learning systems and MLOps workflows.