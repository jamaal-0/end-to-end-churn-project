import pickle
import pandas as pd 
from .pydanticModels import featureColumns
from typing import List

def toDataFrame(data: featureColumns | List[featureColumns]) -> pd.DataFrame:
    #convert the data into a dcitionary 
    #pd.DataFrame([dict]) creates one row where the keys are the cols and the values are the values
    if isinstance(data, list):
        return pd.DataFrame([d.model_dump() for d in data])

    if isinstance(data, featureColumns):
        return pd.DataFrame([data.model_dump()])

    raise TypeError("Invalid input type")


def loadArtifact(name:str):
    name = name.strip().replace(" ","_")
    with open(f"src/artifacts/{name}.pkl", "rb") as f:
        artifact = pickle.load(f)
    return artifact