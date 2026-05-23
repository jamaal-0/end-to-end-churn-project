from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PowerTransformer,
    StandardScaler
)


class dataTransformation:

    def __init__(self):
        pass

   
    def build_preprocessor(self, df_copy):
        df_copy_num = df_copy.select_dtypes(include=['number']).columns.tolist()
        print(df_copy_num)
        df_copy_cat = df_copy.select_dtypes(include=['str']).columns.tolist()
        print(df_copy_cat)


        transformer = []

        #numeric columns
        #scale with simple scaler and transform with yeo-johnson
        if df_copy_num:
            numeric_pipeline = Pipeline([
                ('power', PowerTransformer(method='yeo-johnson')),
                ('scaler', StandardScaler())
            ])

            transformer.append(('numeric', numeric_pipeline, df_copy_num))
        
        #categorcial columns
        #use one hot encodiing 
        if df_copy_cat:
            categorical_pipeline = Pipeline([
                ('encoder', OneHotEncoder())
            ])

            transformer.append(("categorical", categorical_pipeline, df_copy_cat))

    
        return ColumnTransformer(transformer)

    
    def transformData(self, df):

        X = df.drop('Churn', axis=1)
        y = df['Churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        preprocessor = self.build_preprocessor(X_train)
    
        return X_train, X_test, y_train, y_test, preprocessor