# Customer Churn Prediction System

## Overview

This project is an end-to-end machine learning system for customer churn prediction built using Python and scikit-learn. The project covers the complete data science workflow including:

- Exploratory Data Analysis (EDA)
- Feature engineering
- Data preprocessing
- Model training and evaluation
- Hyperparameter tuning
- Threshold optimization
- Model explainability using SHAP
- Modular ML pipeline architecture
- Backend API deployment using FastAPI

The objective of the project is to predict whether a customer is likely to churn based on customer demographics, subscription information, and service usage behaviour.

---

# Dataset

The dataset contains customer information from a telecommunications company.

## Features

| Feature | Description |
|---|---|
| customerID | Unique customer identifier |
| gender | Customer gender |
| SeniorCitizen | Whether customer is a senior citizen |
| Partner | Whether customer has a partner |
| Dependents | Whether customer has dependents |
| tenure | Number of months customer has stayed |
| PhoneService | Whether customer has phone service |
| MultipleLines | Whether customer has multiple lines |
| InternetService | Internet service provider type |
| OnlineSecurity | Online security subscription |
| OnlineBackup | Online backup subscription |
| DeviceProtection | Device protection subscription |
| TechSupport | Technical support subscription |
| StreamingTV | Streaming TV subscription |
| StreamingMovies | Streaming movies subscription |
| Contract | Contract type |
| PaperlessBilling | Whether paperless billing is enabled |
| PaymentMethod | Payment method |
| MonthlyCharges | Monthly subscription charges |
| TotalCharges | Total charges accumulated |
| Churn | Target variable |

---

# Exploratory Data Analysis (EDA)

EDA was performed to:

- Understand feature distributions
- Identify missing values
- Detect class imbalance
- Analyze feature relationships with churn
- Identify predictive features
- Guide preprocessing decisions

## Analysis Performed

### Univariate Analysis

- Distribution plots
- Count plots for categorical variables
- KDE plots for numerical variables

### Bivariate Analysis

Features were analyzed against the target variable (`Churn`) using:

- Box plots
- Count plots grouped by churn status
- KDE plots for numerical features
- Correlation analysis

### Feature Importance Analysis

Mutual Information was used to estimate predictive strength of features.

Top predictive features included:

- Contract
- tenure
- TechSupport
- OnlineSecurity
- InternetService

---

# Feature Engineering

Feature engineering experiments were performed to investigate whether aggregated behavioural features improved predictive performance.

## Engineered Features

### Number of Services Used

Aggregated multiple service-related features into a single engagement metric.

```python
service_cols = [
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies'
]

df['NumServicesUsed'] = (df[service_cols] == 'Yes').sum(axis=1)
```

### No Protection Services Flag

Created a binary feature representing customers with no protection-related services.

```python
df['NoProtectionServices'] = (
    (df['OnlineSecurity'] == 'No') &
    (df['DeviceProtection'] == 'No') &
    (df['TechSupport'] == 'No')
).astype(int)
```

## Outcome

SHAP analysis showed that engineered features did not significantly contribute additional predictive power beyond the original feature set. This indicated that the model was already capturing the majority of the available signal from the raw features.

---

# Data Preprocessing

A preprocessing pipeline was implemented using `ColumnTransformer`.

## Numerical Features

Numerical features were:

- Imputed where necessary
- Scaled using `StandardScaler`

## Categorical Features

Categorical features were:

- Imputed where necessary
- One-hot encoded using `OneHotEncoder`

---

# Model Training

## Baseline Model

A Logistic Regression model was used as the baseline model.

### Baseline Pipeline

```python
Pipeline([
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression())
])
```

---

# Advanced Models

The project also experimented with:

- XGBoost
- Hyperparameter tuning using GridSearchCV

## Hyperparameter Tuning

5-fold cross-validation was used during hyperparameter optimization.

Example parameters tuned:

```python
param_grid = {
    "classifier__n_estimators": [100, 200, 500],
    "classifier__learning_rate": [0.01, 0.05, 0.1],
    "classifier__max_depth": [3, 5, 10],
    "classifier__subsample": [0.6, 0.8, 1.0],
    "classifier__colsample_bytree": [0.6, 0.8, 1.0]
}
```

---

# Model Evaluation

## Metrics Used

The following metrics were used for evaluation:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## Threshold Optimization

Instead of using the default classification threshold of `0.5`, threshold tuning was performed to optimize the precision-recall tradeoff.

Thresholds were evaluated using predicted probabilities from the model.

Example:

```python
thresholds = np.arange(0.1, 0.91, 0.05)
```

This improved recall performance while maintaining acceptable precision.

---

# SHAP Explainability

SHAP values were used to explain model behaviour and identify the most influential features.

## SHAP Analysis Objectives

- Understand feature importance
- Interpret individual predictions
- Validate engineered features
- Identify churn-driving patterns

Key findings:

- Contract type strongly influenced churn probability
- Short tenure customers had significantly higher churn risk
- Customers without support/security services were more likely to churn

---

# Project Architecture

After initial experimentation in the EDA notebook, the project was modularized into a production-style ML pipeline.

## Project Structure

```text
project/
│
├── data_ingestion/
├── data_transformation/
├── model_training/
├── pipeline/
├── api/
├── artifacts/
├── notebooks/
└── README.md
```

---

# Pipeline Components

## 1. Data Ingestion

Responsible for:

- Loading raw data
- Data validation and basic cleaning

---

## 2. Data Transformation

Responsible for:

- Train/test splitting
- Scaling numerical features
- Encoding categorical features
- Constructing preprocessing pipeline

---

## 3. Model Training

Responsible for:

- Model training
- Hyperparameter tuning
- Cross-validation
- Threshold optimization
- Evaluation

---

## 4. Training Pipeline

A pipeline orchestrates the complete workflow:

```text
Data Ingestion
    ↓
Data Transformation
    ↓
Model Training
    ↓
Artifact Serialization
```

---

# Model Serialization

The final trained sklearn pipeline was serialized into reusable model artifacts.

Serialized artifacts included:

- Preprocessing + model pipeline
- Optimized classification threshold
- Model metadata and version information

These artifacts were later loaded by the backend API during application startup to ensure consistent inference behaviour between training and deployment.

A helper-based artifact loading system was implemented to abstract model loading logic from the API endpoints.

---

# Backend Model Initialization

The backend API uses a FastAPI lifespan context manager to load model artifacts once during application startup.

This architecture ensures:

- Models are loaded into memory only once
- Inference remains efficient during runtime
- Prediction endpoints remain lightweight
- Centralized management of deployed model versions

Example:

```python
model = None
threshold = 0.5
name = None

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model, name, threshold

    model_artifact = loadArtifact("LogisticRegressionModel")

    model = model_artifact["model"]
    name = model_artifact["name"]
    threshold = model_artifact.get("threshold", 0.5)

    print("Models loaded successfully at startup")

    yield

    print("API shutting down")
```

The loaded sklearn pipeline internally handles preprocessing and prediction during inference.

---

# Backend API

A backend API was implemented using FastAPI to serve real-time predictions.

## Features

- Single prediction endpoint
- Batch prediction endpoint
- Probability outputs
- Threshold-based churn classification
- Prediction persistence
- Centralized model loading during application startup

---

# Prediction Logging

The backend includes functionality to persist predictions into a local PostgreSQL database.

Each prediction request stores:

- Predicted churn probability
- Final churn classification
- Model version/name used for inference

This enables:

- Prediction auditing
- Inference tracking
- Model monitoring preparation
- Historical prediction analysis

Example:

```python
def save_prediction(probability, churn, model_name):

    conn = get_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO prediction_logs (probability, churn, model_name)
    VALUES (%s, %s, %s)
    """

    cursor.execute(query, (probability, churn, model_name))

    conn.commit()

    cursor.close()
    conn.close()
```

Prediction logging is integrated directly into both single and batch inference endpoints.

---

# API Architecture

The backend loads:

- Serialized model pipeline
- Optimized threshold
- Model metadata

The sklearn pipeline handles preprocessing internally during inference.

```text
Incoming Request
    ↓
DataFrame Conversion
    ↓
Pipeline Preprocessing
    ↓
Model Prediction
    ↓
Threshold Classification
    ↓
JSON Response
```

---

# API Endpoints

## Single Prediction

### Endpoint

```http
POST /predict_single
```

### Response

```json
{
  "churn": true,
  "probability": 0.78,
  "model": "v1"
}
```

---

## Batch Prediction

### Endpoint

```http
POST /predict_batch
```

### Response

```json
{
  "churn": [true, false],
  "probability": [0.81, 0.24],
  "model": "v1"
}
```

---

# Technologies Used

## Machine Learning

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP

## Backend

- FastAPI
- Pydantic
- joblib

## Visualization

- matplotlib
- seaborn

---

# Key Learnings

This project provided practical experience with:

- End-to-end ML workflows
- Feature engineering
- Model explainability
- Threshold optimization
- Production-style ML architecture
- API deployment for ML inference
- Cross-validation and evaluation methodology
- Pipeline-based preprocessing

---

# Future Improvements

Potential future work includes:

- Model monitoring
- Drift detection
- Docker containerization
- CI/CD integration
- Cloud deployment
- Experiment tracking
- Real-time streaming predictions
- Calibration analysis

---

# Conclusion

This project demonstrates the complete lifecycle of a machine learning system from exploratory analysis to deployable API inference.

The final solution combines:

- rigorous EDA
- modular ML engineering
- interpretable modeling
- production-oriented deployment practices

into a single end-to-end churn prediction platform.

