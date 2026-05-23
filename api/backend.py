from fastapi import FastAPI
from .pydanticModels import featureColumns
from .utilis import toDataFrame, loadArtifact
from .predict import predictSingle, predictBatch
from contextlib import asynccontextmanager
from .database import get_connection



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



app = FastAPI(lifespan=lifespan)


@app.post('/predict_single')
def predict_single_endpoint(data:featureColumns):
    data = toDataFrame(data)
    churn, probability = predictSingle(data, model, threshold)
    try:
        save_prediction(probability, churn, name)

    except Exception as e:
        print(f"Save failed: {e}")

    return {
        'churn': churn,
        'probability': probability,
        'threshold': threshold,
        'model': name
    }

@app.post("/predict_batch")
def predict_batch_endpoint(data:list[featureColumns]):
    data = toDataFrame(data)
    churn, probability = predictBatch(data, model, threshold)
    for p, c in zip(probability, churn):
        try:
            save_prediction(float(p), bool(c), name)

        except Exception as e:
            print(f"Save failed: {e}")
    return {
        'churn': churn,
        'probability': probability,
        'threshold': threshold,
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
