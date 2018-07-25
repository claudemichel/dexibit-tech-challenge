import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

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


def build_train_infer(train_X, train_y, val_X, val_y):
    exibition_model = XGBRegressor(n_estimators=XGB_N_ESTIMATOR, learning_rate=XGB_LEARNING_RATE)
    exibition_model.fit(train_X, train_y, early_stopping_rounds=XGB_EARLY_STOPPING_ROUND, eval_set=[(val_X,val_y)])

    val_predictions = exibition_model.predict(val_X)
    return val_predictions


def get_error_rate(train_X, val_X, train_y, val_y):
    val_predictions = build_train_infer(train_X, train_y, val_X, val_y)
    return compute_dexibit_error_rate(val_y, val_predictions)


exibition_train_data = load_csv(exibition_train_file_path)

exibition_pred_data = load_csv(exibition_pred_file_path)

#Define predictors + Drop unrelevant column + Use ont-hot encoding on categorial data
X=pd.get_dummies(exibition_train_data[exibition_predictors].drop(exibition_predictors_to_drop, axis=1))

#Define prediction target: 'ExhibitionVisitors'
y = exibition_train_data.ExhibitionVisitors

#Split data in two groups: validation process data and traininig process data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

error_rate=get_error_rate(train_X, val_X, train_y, val_y)
print("Error rate : %d%%" % (error_rate))

