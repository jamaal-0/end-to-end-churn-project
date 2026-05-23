import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

class ModelTrainer():
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.test_val_split()
         
    
    def model_buidler(self, preprocessor):
        model = Pipeline(steps=[
            ('preprocess', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])

        model.fit(self.X_train_main, self.y_train_main)

        return model
    
    def test_val_split(self): 
        self.X_train_main, self.X_val, self.y_train_main, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, stratify=self.y_train, random_state=42)
        pass
    
    
    def prob_thresh(self, model):
        y_proba = model.predict_proba(self.X_val)[:,1]
        best_threshold = None
        thresholds = np.arange(0.1, 0.91, 0.01)
        best_recall = 0

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            precision = precision_score(self.y_val, y_pred)
            recall = recall_score(self.y_val, y_pred)

            if precision >= 0.6 and recall > best_recall:
                best_recall = recall
                best_threshold = t

        print(best_threshold, best_recall)

        return best_threshold, best_recall
    


    def model_eval(self, model, best_threshold): 
        y_proba = model.predict_proba(self.X_test)[:,1]
        y_pred = (y_proba >= best_threshold).astype(int)
        print(classification_report(self.y_test, y_pred))
        print("ROC-AUC:", roc_auc_score(self.y_test, y_proba))
        
        return y_proba, y_pred
