
import os
import pickle

import pandas as pd
from .dataIngestion import dataIngestion
from .EDA import EDA
from .dataTransformation import dataTransformation
from .modelTrainer import modelTrainer

def save_object(file_path, obj):
    """
    Saves any Python object to a file using pickle.
    Creates directories if they don't exist.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)







#ingest the data 
dataIngestion = dataIngestion()
testPath, trainPath,targetColumn = dataIngestion.savePaths()
print(testPath)
print(trainPath)

'''
#EDA
# Run EDA once
df_train = pd.read_csv(trainPath)
eda = EDA(df_train, targetColumn)
eda.transformsColumns()  # populates column lists
schema = eda.getSchema()
print(schema)
artifact = {
    "name": 'eda_columns',
    "model": schema,
    "pipeline_version": "1.0"
}
save_object("artifacts/eda_columns.pkl", artifact)





#transform the data 
dataTransformation = dataTransformation(targetColumn)
X_train, y_train, X_test, y_test, preprocessor = dataTransformation.transformData(trainPath,testPath, schema)
print(X_train.shape, '\n',y_train.shape, '\n',X_test.shape, '\n',y_test.shape,'\n', end='\n')
artifact = {
    "name": 'Preprocessor',
    "model": preprocessor,
    "pipeline_version": "1.0"
}
save_object("artifacts/preprocessor.pkl", artifact)


#train the models and return the best models for each metric 
modelTrainer = modelTrainer(X_train, y_train, X_test, y_test)
best_model_info = modelTrainer.trainModels()
print(best_model_info)




#save model and preprocessor object *** need to save each model seperately into its own pkl file 

top_name = best_model_info['name']
filename_model = top_name.replace(" ", "_")
top_model = best_model_info['model']
top_threshold = best_model_info['threshold']
artifact = {
    "name": filename_model,
    "model": top_model,
    "threshold": top_threshold,
    "pipeline_version": "1.0"
}
save_object(f"artifacts/{filename_model}.pkl", artifact)





#find a way for the saved preprocessor object to include the EDA or save the EDA to its own .pkl file as well ✅
#save the objects to .pkl files ✅

#create new EDA file and do the universal transformations on it ✅
#refactor the datatransformation file✅


#test backend using postman to validate API ✅
#save predictions in a postgres database ✅

#learn how to read and write to a database

#learn powerBI to do visualisations using the database

#do the visualisations 

#create and save baseline and random models 
#save predictions in a postgres database
#use tableau or powerBI to do visualisations using the database

#uvicorn api.backend:app --reload
#python -m components.pipeline  

'''