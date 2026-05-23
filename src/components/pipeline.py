import os
import pickle
from .dataIngestion import dataIngestion
from .dataTransformation import dataTransformation
from .modelTrainer import ModelTrainer

def save_object(file_path, obj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

#ingest data
dataIngestion = dataIngestion()
df = dataIngestion.read_and_clean()
df_copy = df.copy()

#transform data
dataTransformation = dataTransformation()
X_train, X_test, y_train, y_test, preprocessor = dataTransformation.transformData(df_copy)

#train model
modelTrainer = ModelTrainer(X_train, X_test, y_train, y_test)
model = modelTrainer.model_buidler(preprocessor)
best_threshold, best_recall = modelTrainer.prob_thresh(model)
y_proba, y_pred = modelTrainer.model_eval(model, best_threshold)

#final return should be a model pipeline containing the preprocessor and its best threshold and save those into a .pkl file


artifact = {
    "name": 'LogisticRegression',
    "model": model,
    "threshold": best_threshold,
    "pipeline_version": "1.0"
}

save_object("artifacts/LogisticRegressionModel.pkl", artifact)

       