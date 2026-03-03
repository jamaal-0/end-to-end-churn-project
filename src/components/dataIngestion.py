#read the data 
#split the data into test/train
#save it to the data folder

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os, sys


#dataclass function to save the data to its correct paths
@dataclass
class dataIngestionConfiguration:
    rawPath:str = os.path.join('data','raw.csv')
    trainPath:str = os.path.join('data','train.csv')
    testPath:str = os.path.join('data','test.csv')


#create class for reading the data
class dataIngestion:
    #init should call the dataclass
    def __init__(self):
        self.dataPaths = dataIngestionConfiguration()




 
#one function to save them to their relative paths, should return the test/train paths
    def savePaths(self):
        #read the original data csv
        dataFrame = pd.read_csv('telco_churn.csv')
        targetColumn = 'Churn'

        #inspect the data to understand what dtypes there are 
        print(dataFrame.dtypes)
        duplicates = dataFrame.duplicated().sum()
        print(f'number of duplicates: {duplicates}')
        #create the Data folder to store the CSVs if it does not already exist
        os.makedirs(name=os.path.dirname(self.dataPaths.rawPath), exist_ok=True)
        #store the orignial raw data into rawPath.csv
        dataFrame.to_csv(self.dataPaths.rawPath,index=False,header=True)
        #call test/train/split on the origial raw data and save it into variables

        y = dataFrame["Churn"]               # target

        trainSplit,testSplit = train_test_split(dataFrame,test_size=0.2,stratify=y,random_state=42)
        #call pd.to_csv on those variables to store them into their relative CSVs
        trainSplit:pd.DataFrame
        testSplit:pd.DataFrame
        trainSplit.to_csv(self.dataPaths.trainPath,index=False,header=True)
        testSplit.to_csv(self.dataPaths.testPath,index=False,header=True)
        #return the paths for the training and testing CSVs and the targetColumn
        print('one two')
        return self.dataPaths.testPath, self.dataPaths.trainPath, targetColumn
    



if __name__=='__main__':
    dataIngestion = dataIngestion()
    testPath, trainPath,targetColumn = dataIngestion.savePaths()
    print(testPath)
    print(trainPath)
    print(targetColumn)