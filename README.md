# end-to-end-churn-project
Overview

This project is an end-to-end applied machine learning system designed to predict customer churn probability using structured customer behavioural data.

The system covers the full ML lifecycle including:

• Data ingestion and preprocessing

• Exploratory data analysis (EDA)

• Model training and evaluation

• Pipeline serialization

• Production inference serving

•	Database prediction logging

The project demonstrates practical deployment-oriented machine learning engineering.

⸻

Architecture

	Raw Dataset
		↓
	Data Cleaning & Feature Engineering
		↓
	Exploratory Data Analysis (EDA)
		↓
	Model Training & Validation
		↓
	Pipeline Serialization (Model + Preprocessor)
		↓
	FastAPI Inference Service
		↓
	Prediction Logging
		↓
	PostgreSQL Storage

⸻

Technologies Used

Programming
	•	Python 3.13

Machine Learning
	•	Scikit-learn
	•	NumPy
	•	Pandas

Models (Experimented)
	•	Ensemble tree-based classifiers

Deployment
	•	FastAPI
	•	Uvicorn

Database
	•	PostgreSQL
	•	psycopg2 driver

⸻

Dataset

The project uses customer behavioural features including:
	•	Demographic attributes
	•	Service usage information
	•	Account contract information
	•	Billing preferences

Target variable:
	•	Churn classification (Yes / No)

⸻

Data Processing Pipeline

The preprocessing pipeline is implemented using ColumnTransformer and includes:

Numeric Feature Processing
	•	Missing value imputation
	•	Yeo-Johnson power transformation for skew reduction
	•	Feature scaling

Categorical Feature Processing
	•	Most-frequent imputation
	•	One-hot encoding with unknown category handling

⸻

Model Training Strategy

Multiple classification models were evaluated.

The optimal model was selected using:
	•	Validation performance
	•	F1-score optimization
	•	Threshold calibration

Decision threshold was stored as model artifact.

⸻

Production Inference System

The project implements a RESTful inference API using FastAPI.

Endpoints include:
	•	Single prediction inference
	•	Batch prediction inference

The inference pipeline loads serialized artifacts:
	•	Trained model
	•	Preprocessing transformer
	•	Prediction threshold

⸻

Database Logging System

Prediction results are automatically stored in PostgreSQL.

Logged information includes:
	•	Prediction probability
	•	Binary churn decision
	•	Model identifier
	•	Timestamp

This enables monitoring and future analytical processing.

⸻

API Endpoints

POST /predict_single

Accepts customer feature JSON input and returns:
	•	Churn decision
	•	Prediction probability
	•	Model name

⸻

POST /predict_batch

Accepts batch feature arrays and returns:
	•	Batch churn predictions
	•	Probability scores

All predictions are persisted in database storage.

⸻

System Design Principles

The project follows production-style ML design considerations:
	•	Separation of training and inference pipelines
	•	Artifact-based model serving
	•	Explicit threshold decisioning
	•	Schema-aware preprocessing
	•	Type safety validation

⸻

Future Work

Planned extensions include:
	•	Dashboard visualization integration (Power BI / Tableau)
	•	Model drift monitoring
	•	Asynchronous prediction logging
	•	Containerized deployment
	•	Automated retraining triggers

⸻

Key Contributions of This Project
	•	End-to-end machine learning lifecycle implementation
	•	Deployment-ready inference API
	•	Database-backed prediction tracking
	•	Pipeline serialization strategy
	•	Business-oriented classification calibration

⸻

Author

Jamaal
Applied Machine Learning & Distributed AI
