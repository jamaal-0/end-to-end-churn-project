import pickle
from fastapi import FastAPI
from .pydanticModels import featureColumns
from .utilis import toDataFrame, loadArtifact
from .predict import predictSingle, predictBatch
from contextlib import asynccontextmanager
from .database import get_connection

# -----------------------------
# Global Cache Variables
# -----------------------------

model = None
preprocessor = None
threshold = 0.5
name = None

# -----------------------------
# Lifespan Manager
# -----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model, preprocessor, name, threshold

    model_artifact = loadArtifact("XGBClassifier")
    model = model_artifact["model"]
    name = model_artifact["name"]
    threshold = model_artifact.get("threshold", 0.5)

    preprocessor_artifact = loadArtifact("preprocessor")
    preprocessor = preprocessor_artifact["model"]

    schema_artifact = loadArtifact("eda_columns")
    schema = schema_artifact['model']

    print("✅ Models loaded successfully at startup")

    yield

    print("🧠 API shutting down")


# -----------------------------
# FastAPI App
# -----------------------------

app = FastAPI(lifespan=lifespan)


@app.post('/predict_single')
def predict_single_endpoint(data:featureColumns):
    data = toDataFrame(data)
    churn, probability = predictSingle(data, model, preprocessor, threshold)
    save_prediction(probability, churn, name)
    return {
        'churn': churn,
        'probability': probability,
        'model': name
    }

@app.post("/predict_batch")
def predict_batch_endpoint(data:list[featureColumns]):
    data = toDataFrame(data)
    churn, probability = predictBatch(data, model, preprocessor, threshold)
    for p, c in zip(probability, churn):
        save_prediction(float(p), bool(c), name)
    return {
        'churn': churn,
        'probability': probability,
        'model': name
    }

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
