import pandas as pd

class dataIngestion:
    def __init__(self):
        pass
        

    def read_and_clean(self):
        #read the original data csv
        df = pd.read_csv('/Users/v/Documents/dataScienceProject/src/telco_churn.csv')

        #do safe basic cleaning on raw data
        df = self.basic_cleaning(df)
        
        return df
    

    def basic_cleaning(self, df):
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.drop(columns=['customerID'])
        df = df.dropna()
        df['Churn'] = df['Churn'].map({'No':0, 'Yes':1})
        return df
     