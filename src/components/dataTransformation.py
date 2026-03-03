import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler
)


class dataTransformation:

    def __init__(self, targetColumn: str):
        self.targetColumn = targetColumn

    # ------------------------------
    # Preprocessor Builder
    # ------------------------------
    def preprocess(self, numericColumns, catColumns):

        transformers = []

        # ---- Numeric pipelines ----

        
        if numericColumns:
            skewedNumericPipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("power", PowerTransformer(method="yeo-johnson")),
                ("scaler", StandardScaler())
            ])

            transformers.append(
                ("num_skewed", skewedNumericPipeline, numericColumns)
            )

        # ---- Categorical pipelines ----

        if catColumns:
            normalCatPipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ))
            ])

            transformers.append(
                ("cat_normal", normalCatPipeline, catColumns)
            )

        

        return ColumnTransformer(transformers)

    # ------------------------------
    # Data Transformation Pipeline
    # ------------------------------
    def transformData(self, trainPath, testPath, schema):

        train_df = pd.read_csv(trainPath)
        test_df = pd.read_csv(testPath)

        train_df = train_df.copy()
        test_df = test_df.copy()

        #train_df['TotalCharges'] = pd.to_numeric(train_df['TotalCharges'], errors='coerce')
        #test_df['TotalCharges'] = pd.to_numeric(test_df['TotalCharges'], errors='coerce')


        

        # ---- Schema unpack ----
        print(test_df.shape)

        drop_columns = schema["drop_columns"]

        numericColumns = (
            schema["skewedNumericCols"] +
            schema["normalNumericCols"] +
            schema["outliersNumericCols"] +
            schema["skew_outlier_cols"]
        )

        catColumns = (
            schema["normalCategoricalCols"] +
            schema["imbalancedcategoricalCols"]
        )


        # ---- Separate target ----

        y_train = train_df[self.targetColumn].map({'No': 0, 'Yes': 1}).astype(int).values
        y_test = test_df[self.targetColumn].map({'No': 0, 'Yes': 1}).astype(int).values

        train_df = train_df.drop(columns=[self.targetColumn], errors="ignore")
        test_df = test_df.drop(columns=[self.targetColumn], errors="ignore")

        # ---- Drop schema columns ----
        print(drop_columns)

        train_df = train_df.drop(columns=drop_columns, errors="ignore")
        test_df = test_df.drop(columns=drop_columns, errors="ignore")
        print(train_df.columns)

        # ---- Numeric coercion ----

        for df in [train_df, test_df]:
            if "TotalCharges" in df.columns:
                df["TotalCharges"] = pd.to_numeric(
                    df["TotalCharges"],
                    errors="coerce"
                )

        # ---- Build pipeline ----

        preprocessor = self.preprocess(
            numericColumns,
            catColumns
        )

        # ---- Fit transform training data ----

        X_train = preprocessor.fit_transform(train_df)
        X_test = preprocessor.transform(test_df)

        # ---- Ensure numpy array safety ----

        X_train = np.atleast_2d(X_train)
        X_test = np.atleast_2d(X_test)

        print(X_train, '\n',y_train, '\n',X_test, '\n',y_test,'\n', end='\n')
        

        return X_train, y_train, X_test, y_test, preprocessor