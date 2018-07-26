import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from matplotlib import pyplot as plt

#TODO: Create a constant or config file
exibition_train_file_path = '../ressources/anon-exhibition_train-input.csv'
exibition_pred_file_path = '../ressources/anon-exhibition_pred-input.csv'

#TODO: Find a more elegant way to get column name except exibition visitors (ex:  kaggle - handle missing value)
exibition_predictors=['ExhibitionName', 'Date', 'DaysRun', 'DaysLeft', 'TotalDuration',
       'PercentComplete', 'Year', 'Season', 'Month', 'WeekDay', 'ArtistName',
       'CategoryName', 'LocationName', 'MarketingSpend', 'TicketPrice', 'VenueVisitors', 'ConversionRate']
# exibition_predictors_to_drop = ['Date', 'Year', 'Month', 'DaysRun', 'DaysLeft', 'PercentComplete','ArtistName', 'ConversionRate']
exibition_predictors_to_drop = ['Date', 'ConversionRate']

XGB_N_ESTIMATOR = 1000
XGB_EARLY_STOPPING_ROUND = 5
XGB_LEARNING_RATE = 0.05

#Def load data
def load_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except:
        print("Opps! Invalid file. Please check file location")


def compute_dexibit_error_rate(val_y, val_predictions):
    val_y, val_predictions = np.array(val_y), np.array(val_predictions)
    return (1 - np.mean(np.abs((val_y-val_predictions)/val_y)))*100


def compute_score_from_mae(scores):
    return -1*scores.mean()


def build_train_infer(train_X, train_y, val_X, val_y):
    exibition_model = XGBRegressor(n_estimators=XGB_N_ESTIMATOR, learning_rate=XGB_LEARNING_RATE)
    exibition_model.fit(train_X, train_y, early_stopping_rounds=XGB_EARLY_STOPPING_ROUND, eval_set=[(val_X,val_y)])

    val_predictions = exibition_model.predict(val_X)
    return val_predictions


def get_error_rate(X,y,fold_number):
    pipeline = make_pipeline(XGBRegressor())
    cv = KFold(n_splits=fold_number,shuffle=False)
    predictions = cross_val_predict(pipeline, X, y, cv=cv)
    return compute_dexibit_error_rate(y,predictions)

def display_cross_validation_plot():
    plt.plot(fold_numbers, error_rate_evolution)
    plt.xlabel("Cross-validation - Folds value")
    plt.ylabel("Error rate (%)")
    plt.show()


exibition_train_data = load_csv(exibition_train_file_path)

exibition_pred_data = load_csv(exibition_pred_file_path)

#Define predictors + Drop unrelevant column + Use ont-hot encoding on categorial data
X=pd.get_dummies(exibition_train_data[exibition_predictors].drop(exibition_predictors_to_drop, axis=1))

#Define prediction target: 'ExhibitionVisitors'
y = exibition_train_data.ExhibitionVisitors

error_rate_evolution = list()
fold_numbers=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

for fold_number in fold_numbers:
    error_rate=get_error_rate(X,y, fold_number)
    print("Fold number %d - Error rate :%d"%(fold_number,error_rate))
    error_rate_evolution.append(error_rate)


display_cross_validation_plot()
