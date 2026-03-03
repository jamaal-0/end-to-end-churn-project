import numpy as np 
import pandas as pd
from components.dataIngestion import dataIngestion


#the goal of the EDA is to inspect the data and decide what transformations to apply to which columns 


class EDA:
    def __init__(self, data:pd.DataFrame, targetColumn):
        self.data = data.copy()
        self.targetColumn = targetColumn

        self.skewedNumericCols = []
        self.outliersNumericCols = []
        self.skew_outlier_cols = []
        self.normalNumericCols = []

        self.normalCategoricalCols = []
        self.imbalancedcategoricalCols = []

        self.drop_columns = []


    def missingValues(self):
        #idenfity if there are missing values in the data
        missingDict = {}
        for col in self.data.columns:
            missing = self.data[col].isna().sum()
            #for the cols with missing values add them to a dict such that col:num of missing
            if missing > 0:
                missingDict[col] = missing
        #return the dict
        return missingDict
    
    #inspect the columns dtypes to fix and errors, create the numeric and categorical columns
    def createColumns(self):
        #print(self.data.dtypes)
        self.data['TotalCharges'] = pd.to_numeric(self.data['TotalCharges'], errors='coerce')
        self.data.drop(columns=self.targetColumn, inplace=True)
        self.rawNumeric = self.data.select_dtypes(include=['number']).columns.tolist()
        self.rawCat = self.data.select_dtypes(include=['object']).columns.tolist()
        
        

##
        #remove the dicts for skewed data and class imbalance 
        #for cardinalitycheck just fix the lists
        #for insepct distribution collect the skews for each numeric feature in a list

##

    #clean the numeric columns of the low cardinality features
    # if a feeature is truely numeric then seperate the skewed and normal  columns 
    def cardinalityCheck(self):
        for col in self.rawNumeric[:]:
            if len(self.data[col].unique()) < 10:
                self.data[col] = self.data[col].astype('category')
                self.rawNumeric.remove(col)
                self.rawCat.append(col)
                #print(f"{col} is not continuous — check balance instead")

        
        
    

        
    #get the class imabalance for categorical columns     
    def classImbalance(self, col, rare_thresh=0.05, dominance_factor=2):
        categoricalClassImbalance = {}

    
        dist = self.data[col].value_counts(normalize=True)
        categoricalClassImbalance[col] = dist.to_dict()

        max_ratio = dist.max()
        min_ratio = dist.min()
        n_classes = len(dist)

        uniform_ratio = 1 / n_classes

        dominance = max_ratio > dominance_factor * uniform_ratio
        rare_class = min_ratio < rare_thresh

        return dominance, rare_class

    

    #figure out which columms contain outliers using the IQR
    def handleOutliers(self,col):

        Q1 = self.data[col].quantile(0.25)
        Q3 = self.data[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = self.data[(self.data[col] < lower) | (self.data[col] > upper)]
        return outliers


    #find out which collumns are closely correlated or provide no useful information
    def cleanUselessFeatures(self,correlationThreshold = 0.9, missing_threshold = 0.8):
        correlatedColumns = []
        high_missing_cols = []
        constant_cols = []

        ###determining correlation 
        for col in self.rawNumeric:
            if self.data[col].nunique() <= 1:
                constant_cols.append(col)
        
            

        #drop constant columns because correlation with a constant column is meaningless (always NaN or 1)
        numeric_df = self.data[self.rawNumeric].drop(columns=constant_cols)
        #obtain correlation matrix (correlation, whether +ve or -ve, is the same)
        corrMatrix = numeric_df.corr().abs()
        #np.triu obtains a upper triangle matrix
        #np.ones is used to be able to cast as bool
        #k=1 means start 1 above the leading diagonal 
        mask = np.triu(np.ones(corrMatrix.shape), k=1).astype(bool)
        #.where replace values where the condition is False.
        upperTri = corrMatrix.where(mask)
        for col in upperTri.columns:
            to_drop = [c for c in upperTri.index if upperTri.loc[c,col]>=correlationThreshold]
            if to_drop:
                correlatedColumns.extend(to_drop)
        correlatedColumns = list(set(correlatedColumns))

        
        #columns with too many missing values
        for col in self.data.columns:
            missing_ratio = self.data[col].isna().mean()
            if missing_ratio > missing_threshold:
                high_missing_cols.append(col)
        


        
        self.drop_columns = list(set(correlatedColumns + high_missing_cols + constant_cols + ['customerID']))

    
    def removeDroppedColumns(self):
        if not hasattr(self, 'drop_columns'):
            return

        # Convert to set for faster lookup
        drop_set = set(self.drop_columns)

        # Filter each feature list
        self.rawCat       = [c for c in self.rawCat if c not in drop_set]
        self.rawNumeric       = [c for c in self.rawNumeric if c not in drop_set]


        



    def transformsColumns(self):
        #create the categorical and numeric columns 
        self.createColumns()

        #check the cardinality of numeric columns and to type cast to categorical if needed 
        self.cardinalityCheck()

        # Determine drop columns first
        self.cleanUselessFeatures()

        # Remove them from all feature lists
        self.removeDroppedColumns()

        #numeric  columns
        for col in self.rawNumeric:
            # detect skew
            skew = self.data[col].skew()
            if abs(skew) >(0.5):
                self.skewedNumericCols.append(col)
            #detect outlier
            outliers = self.handleOutliers(col)
            if len(outliers) > 0:
                self.outliersNumericCols.append(col)
        
        
        #categorical columns 
        for col in self.rawCat:
            dominance, rare_class = self.classImbalance(col, rare_thresh=0.05, dominance_factor=2)
            if dominance or rare_class:
                self.imbalancedcategoricalCols.append(col)
            else:
                self.normalCategoricalCols.append(col)



        #seperate into correct arrays

        for col in self.rawNumeric:
        # Col in both skewed and outliers
            if col in self.skewedNumericCols and col in self.outliersNumericCols:
                self.skew_outlier_cols.append(col)
                self.skewedNumericCols.remove(col)
                self.outliersNumericCols.remove(col)

            # Col in neither skewed nor outliers → normal numeric
            elif col not in self.skewedNumericCols and col not in self.outliersNumericCols:
                self.normalNumericCols.append(col)

        for col in self.rawCat:
            if col in self.imbalancedcategoricalCols and col in self.normalCategoricalCols:
                self.normalCategoricalCols.remove(col)

        
        #cast categorical cols to categorty dtype
        for col in self.imbalancedcategoricalCols:
            self.data[col] = self.data[col].astype('category')

        for col in self.normalCategoricalCols:
            self.data[col] = self.data[col].astype('category')

        
        

                


    def getSchema(self):
        return {
            "normalNumericCols": self.normalNumericCols,
            "skewedNumericCols": self.skewedNumericCols,
            "outliersNumericCols": self.outliersNumericCols,
            "skew_outlier_cols": self.skew_outlier_cols,
            "normalCategoricalCols": self.normalCategoricalCols,
            "imbalancedcategoricalCols": self.imbalancedcategoricalCols,
            "drop_columns": self.drop_columns
        }
        


if __name__=='__main__':
    dataIngestion = dataIngestion()
    testPath, trainPath,targetColumn = dataIngestion.savePaths()
    data = pd.read_csv(trainPath)
    eda = EDA(data)
    (normalNumericCols,
                skewedNumericCols,
                outliersNumericCols,
                skew_outlier_cols,
                normalCategoricalCols,
                imbalancedcategoricalCols) = eda.transformsColumns()
    
    print("\nNormal numeric:")
    print(normalNumericCols)
    print(data[normalNumericCols].dtypes)

    print("\nSkewed numeric:")
    print(skewedNumericCols)
    print(data[skewedNumericCols].dtypes)

    print("\nOutlier numeric:")
    print(outliersNumericCols)
    print(data[outliersNumericCols].dtypes)

    print("\nSkew + outlier numeric:")
    print(skew_outlier_cols)
    print(data[skew_outlier_cols].dtypes)

    print("\nNormal categorical:")
    print(normalCategoricalCols)
    print(data[normalCategoricalCols].dtypes)

    print("\nImbalanced categorical:")
    print(imbalancedcategoricalCols)
    print(data[imbalancedcategoricalCols].dtypes)
        