import pandas as pd 



def predictSingle(data: pd.DataFrame, model, threshold):

    probability = model.predict_proba(data)[0, 1]

    churn = probability >= threshold

    return bool(churn), float(probability)


def predictBatch(data: pd.DataFrame, model, threshold):

    probability = model.predict_proba(data)[:, 1]

    churn = probability >= threshold

    return churn.tolist(), probability.tolist()
        

        

    

    

    