import pickle
from fastapi import HTTPException
import numpy as np
import pandas as pd 
from .pydanticModels import featureColumns
from .utilis import loadArtifact

#global cache
#only to remove the yellow lines



def predictSingle(data:pd.DataFrame, model, preprocessor, threshold):

        

        #feed data to preprocessor object 
        dataArray = preprocessor.transform(data)
        #if dataArray.ndim != 2:
        #      raise ValueError(f"Expected 2D array, got {dataArray.ndim}D")
        


        #feed data to a specific model
        probability = model.predict_proba(dataArray)[0, 1]

        churn = probability >= threshold
        
        return bool(churn), float(probability)


def predictBatch(data:pd.DataFrame, model, preprocessor, threshold):


        #feed data to preprocessor object 
        dataArray = preprocessor.transform(data)
        #if dataArray.ndim != 2:
        #        raise ValueError(f"Expected 2D array, got {dataArray.ndim}D")
        
        
        

        #feed data to a specific model
        probability = model.predict_proba(dataArray)[:, 1]

        churn = probability >= threshold

        return churn.tolist(), probability.tolist()
        

        

    

    

    