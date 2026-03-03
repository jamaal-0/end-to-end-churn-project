


import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class modelTrainer():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.trainArray_X, self.trainArray_y = X_train, y_train
        self.testArray_X, self.testArray_y = X_test, y_test
        self.validationSet()


    def getModels(self):
        models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "XGBClassifier": XGBClassifier(eval_metric="logloss"),
            "CatBoost Classifier": CatBoostClassifier(verbose=False),
            "AdaBoost Classifier": AdaBoostClassifier(),
        }

        params = {
            "Random Forest": {
                "n_estimators": [100, 200, 500],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None]
            },
            "Decision Tree": {
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": [None, "sqrt", "log2"]
            },
            "Gradient Boosting": {
                "n_estimators": [100, 200, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 10],
                "subsample": [0.6, 0.8, 1.0],
                "min_samples_split": [2, 5, 10]
            },
            "Logistic Regression": {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs"]
            },
            "XGBClassifier": {
                "n_estimators": [100, 200, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 10],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0]
            },
            "CatBoost Classifier": {
                "iterations": [100, 200, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "depth": [3, 5, 10],
                "l2_leaf_reg": [1, 3, 5]
            },
            "AdaBoost Classifier": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
            }
        }

        return models, params
    
    def validationSet(self):

        self.trainArray_X, self.X_val, self.trainArray_y, self.y_val = train_test_split(
            self.trainArray_X, self.trainArray_y,
            test_size=0.2,
            stratify=self.trainArray_y,
            random_state=42
            )
        
    def findOptimalThreshold(self, best_model):

        val_probs = best_model.predict_proba(self.X_val)[:, 1]

        precision, recall, thresholds = precision_recall_curve(self.y_val, val_probs)

        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)

        best_index = np.argmax(f1_scores)
        best_threshold = thresholds[best_index]

        print("Optimal threshold:", best_threshold)
        return best_threshold
    
    def trainModels(self):
        trained_models = {}
        #train the model
        modelsDict,paramsDict = self.getModels()

        for name, model in modelsDict.items():
            paramGrid = paramsDict[name]
            print(name)

            #do cross validation for hyperparameter tuning 
            if paramGrid and "CatBoost" not in name:  # Only do GridSearchCV if there are hyperparameters
                print('started')
                gridSearch = GridSearchCV(model, paramGrid, cv=5, scoring="f1", n_jobs=-1)
                #gridSearch = RandomizedSearchCV(model, paramGrid, n_iter=10, cv=5)
                print('.fit')
                gridSearch.fit(self.trainArray_X, self.trainArray_y)
                best_model = gridSearch.best_estimator_
                print('ended')
            else:
                print('started')
                # Just fit the model directly if no hyperparameters
                model.fit(self.trainArray_X, self.trainArray_y)
                best_model = model
                print('ended') 
            trained_models[name] = best_model

            


        #find each model's best threshold 
        best_val_f1 = 0
        top_name, top_model, top_threshold = None, None, None
        for name, best_model in trained_models.items():
            threshold = self.findOptimalThreshold(best_model)
            final_preds = (best_model.predict_proba(self.X_val)[:,1] >= threshold).astype(int)
            f1 = f1_score(self.y_val, final_preds)
            if f1 > best_val_f1:
                best_val_f1 = f1
                top_name = name
                top_model = best_model
                top_threshold = threshold
        
        best_model_info = {'name':top_name, 'model':top_model, 'threshold':top_threshold}
        return best_model_info
    '''
    
        

            
            










            #give predictions 
            #y_train_pred = best_model.predict(self.trainArray_X)
            #y_test_pred = best_model.predict(self.testArray_X)
            

            #test metrics
            #




            #create a function for each metric that returns the best model for that metric 
            #input to the funcion is all the models' y_prediction, the true y values and the models' name
            #check based off of the current max or min 
            #return the name of the model and obtain the instance
            #in that
            # for each metric i want the best model
            #for metric in metrics 
            





            # Summary metrics
            #performance_summary[name] = self.evaluateModelTrain(y_train_pred)
        #accuracy_model, accuracy_value = self.eval_accuracy(trained_models)
        #precision_model, precision_value = self.eval_precision(trained_models)
        #roc_auc_model, roc_auc_value = self.eval_roc_auc(trained_models)
        f1_model, f1_value = self.eval_f1(trained_models)
        #recall_model, recall_value = self.eval_recall(trained_models)

        best_model_info = {'model_name': f1_model,'value': f1_value, 'model_object': trained_models[f1_model]}

        best_model_info['threshold'] = self.findOptimalThreshold(best_model_info['model_object'])





        #models_list = [accuracy_model,precision_model,roc_auc_model,f1_model,recall_model]
        #metrics = ['accuracy', 'precision', 'roc_auc', 'f1', 'recall']
        #values_list = [accuracy_value,precision_value,roc_auc_value,f1_value,recall_value]

        #for i in range(len(metrics)):
        #    best_per_metric[metrics[i]] = {'model_name': models_list[i],'value': values_list[i], 'model_object': trained_models[models_list[i]]}
        
        #print(best_per_metric)
        
        return best_model_info
        '''
    '''


        
        
        
        
        

        print(performance_summary)
        metrics = ['rmse', 'r2', 'mae', 'mse']

        for metric in metrics:
            if metric in ['rmse', 'mae', 'mse']:
                best_model_name = min(performance_summary, key=lambda k: performance_summary[k][metric])
            else:
                # Higher is better for R²
                best_model_name = max(performance_summary, key=lambda k: performance_summary[k][metric])
            
            best_models_per_metric[metric] = {
                'model': best_model_name,
                'model_object': trained_models[best_model_name],
                'value': performance_summary[best_model_name][metric]
            }
        print(best_models_per_metric)
        return best_models_per_metric
        '''
    
    #create a function for each metric that returns the best model for that metric 
    #input to the funcion is all the models' y_prediction, the true y values and the models' name
    #check based off of the current max or min 
    #return the name of the model and obtain the instance
    #in that
    # for each metric i want the best model
    #for metric in metrics 
    '''
    # Accuracy
    def eval_accuracy(self, trained_models):
        true_y = self.testArray_y
        acc = {}
        for model_name, model in trained_models.items():
            pred_y = model.predict(self.testArray_X)
            accuracy = accuracy_score(y_true=true_y, y_pred=pred_y)
            acc[model_name] = accuracy
        best_model = max(acc, key=acc.get)
        return best_model, acc[best_model]

    # ROC-AUC
    def eval_roc_auc(self, trained_models):
        true_y = self.testArray_y
        auc_scores = {}
        for model_name, model in trained_models.items():
            # use predict_proba if available, fallback to predict for regressors
            if hasattr(model, "predict_proba"):
                pred_prob = model.predict_proba(self.testArray_X)[:,1]
            else:
                pred_prob = model.predict(self.testArray_X)
            auc = roc_auc_score(true_y, pred_prob)
            auc_scores[model_name] = auc
        best_model = max(auc_scores, key=auc_scores.get)
        return best_model, auc_scores[best_model]

    # Precision
    def eval_precision(self, trained_models):
        true_y = self.testArray_y
        precisions = {}
        for model_name, model in trained_models.items():
            pred_y = model.predict(self.testArray_X)
            precisions[model_name] = precision_score(true_y, pred_y, zero_division=0)
        best_model = max(precisions, key=precisions.get)
        return best_model, precisions[best_model]

    # Recall
    def eval_recall(self, trained_models):
        true_y = self.testArray_y
        recalls = {}
        for model_name, model in trained_models.items():
            pred_y = model.predict(self.testArray_X)
            recalls[model_name] = recall_score(true_y, pred_y, zero_division=0)
        best_model = max(recalls, key=recalls.get)
        return best_model, recalls[best_model]

    # F1 Score
    def eval_f1(self, trained_models):
        true_y = self.testArray_y
        f1_scores = {}
        for model_name, model in trained_models.items():
            pred_y = model.predict(self.testArray_X)
            f1_scores[model_name] = f1_score(true_y, pred_y, zero_division=0)
        best_model = max(f1_scores, key=f1_scores.get)
        return best_model, f1_scores[best_model]




    
    def evaluateModelTrain(self, y_train_pred):
        return {
            "tr2": r2_score(self.trainArray_y, y_train_pred),
            "train_mae" : mean_absolute_error(self.trainArray_y, y_train_pred),
            'train_r2_square' : r2_score(self.trainArray_y, y_train_pred),
            'train_mse' : mean_squared_error(self.trainArray_y, y_train_pred),
            'train_rmse' : np.sqrt(mean_squared_error(self.trainArray_y, y_train_pred)),
        }
    

    def evaluateModel(self, true_y, y_pred):
        return {
            'rmse' : np.sqrt(mean_squared_error(true_y, y_pred)),
            "r2": r2_score(true_y, y_pred),
            'mae' : mean_absolute_error(true_y, y_pred),
            'mse' : mean_squared_error(true_y, y_pred),
        }
    '''

        
        
        
