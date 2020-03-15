import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import pickle
from datetime import datetime
import hashlib
import binascii
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from flask import Blueprint, request, jsonify, send_from_directory

from app_modeling_web.config import (
    API_KEY_CROSS_VAL_NUM, API_KEY_DATADIR, 
    API_KEY_MODEL_PARAMETER, API_KEY_RANDOM_STATE,
    API_KEY_RESULT_CV_SCORE, API_KEY_RESULT_VALIDATION_SET_SCORE, 
    API_KEY_RESULT_BEST_PARAM, API_KEY_RESULT_TEST_SCORE, 
    API_KEY_RESULT_MSE, API_KEY_RESULT_RMSE, API_KEY_RESULT_RMSE_MEAN, 
    API_KEY_RESULT_MEAN_FIT_TIME, API_KEY_RESULT_STD_FIT_TIME,
    API_KEY_RESULT_MEAN_SCORE_TIME, API_KEY_RESULT_STD_SCORE_TIME, 
    API_KEY_RESULT_PARAMS, API_KEY_RESULT_MEAN_TEST_SCORE, 
    API_KEY_RESULT_STD_TEST_SCORE, API_KEY_RESULT_RANK_TEST_SCORE, 
    API_KEY_DATA_COLUMNS,
    GRID_PARAM_ENET_ALPHA, GRID_PARAM_ENET_L1RATIO, GRID_PARAM_LASSO_ALPHA, 
    GRID_PARAM_POLY_DEG, GRID_PARAM_RIDGE_ALPHA,
    MODEL_PARAM_ALPHA, MODEL_PARAM_CV, MODEL_PARAM_DEG, 
    MODEL_PARAM_FIT_INTERCEPT, MODEL_PARAM_L1_RATIO, MODEL_PARAM_MODEL_TYPE, 
    MODEL_PARAM_MAXITER, MODEL_PARAM_X_COLUMNS, 
    MODEL_PARAM_Y_COLUMN, MODEL_TYPE_ELNET, MODEL_TYPE_LASSO, MODEL_TYPE_RIDGE, 
    PIPE_KEY_POLY, PIPE_KEY_SCALAR,
    PREFIX_API, URL_INDEX_API)


api = Blueprint(PREFIX_API, __name__)


def generate_data(dat, X_columns, y_column, phase='validation', test_size=0.3, random_state=0):
    """Generate training data set

    Args:
        dat (pd.DataFrame):
        X_columns (list):
        y_columns (list):
        phase (str): validation / test
        test_size (float):
        random_state (int):

    Return:
        ret (tuple): X_train, X_test, y_train, y_test
    """
        
    X = dat[X_columns]
    y = dat[y_column]
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, 
                                            test_size=test_size, random_state=random_state)

    if phase == 'test':
        return (X_trainval, y_trainval, X_test, y_test)

    if phase == 'validation':
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, 
                                                test_size=test_size, random_state=random_state)
        return (X_train, y_train, X_val, y_val)

def set_model_validation(param_dict):
    if param_dict[MODEL_PARAM_MODEL_TYPE] == 'ridge':
        model = Ridge(
            max_iter=param_dict[MODEL_PARAM_MAXITER], 
            fit_intercept=param_dict[MODEL_PARAM_FIT_INTERCEPT])
        grid_param = {
            GRID_PARAM_POLY_DEG: param_dict[MODEL_PARAM_DEG],
            GRID_PARAM_RIDGE_ALPHA: param_dict[MODEL_PARAM_ALPHA]
            }

    elif param_dict[MODEL_PARAM_MODEL_TYPE] == 'lasso':
        model = Lasso(
            max_iter=param_dict[MODEL_PARAM_MAXITER], 
            fit_intercept=param_dict[MODEL_PARAM_FIT_INTERCEPT])
        grid_param = {
            GRID_PARAM_POLY_DEG: param_dict[MODEL_PARAM_DEG],
            GRID_PARAM_LASSO_ALPHA: param_dict[MODEL_PARAM_ALPHA]
            }

    elif param_dict[MODEL_PARAM_MODEL_TYPE] == 'elasticnet':
        model = ElasticNet(
            l1_ratio=param_dict[MODEL_PARAM_L1_RATIO],
            max_iter=param_dict[MODEL_PARAM_MAXITER],
            fit_intercept=param_dict[MODEL_PARAM_FIT_INTERCEPT])
        grid_param = {
            GRID_PARAM_POLY_DEG: param_dict[MODEL_PARAM_DEG],
            GRID_PARAM_ENET_ALPHA: param_dict[MODEL_PARAM_ALPHA],
            GRID_PARAM_ENET_L1RATIO: param_dict[MODEL_PARAM_L1_RATIO]
            }
    else:
        return None, None
    pipe = Pipeline([
        ('scalar', StandardScaler()), 
        ('poly', PolynomialFeatures()),
        (param_dict[MODEL_PARAM_MODEL_TYPE], model)
    ])
    return pipe, grid_param

def set_model_test(param_dict):
    if param_dict[MODEL_PARAM_MODEL_TYPE] == 'ridge':
        model = Ridge(
            alpha=param_dict[MODEL_PARAM_ALPHA],
            max_iter=param_dict[MODEL_PARAM_MAXITER], 
            fit_intercept=param_dict[MODEL_PARAM_FIT_INTERCEPT])

    elif param_dict[MODEL_PARAM_MODEL_TYPE] == 'lasso':
        model = Lasso(
            alpha=param_dict[MODEL_PARAM_ALPHA],
            max_iter=param_dict[MODEL_PARAM_MAXITER], 
            fit_intercept=param_dict[MODEL_PARAM_FIT_INTERCEPT])

    elif param_dict[MODEL_PARAM_MODEL_TYPE] == 'elasticnet':
        model = ElasticNet(
            alpha=param_dict[MODEL_PARAM_ALPHA],
            l1_ratio=param_dict[MODEL_PARAM_L1_RATIO],
            max_iter=param_dict[MODEL_PARAM_MAXITER],
            fit_intercept=param_dict[MODEL_PARAM_FIT_INTERCEPT])
    else:
        return None
    pipe = Pipeline([
        ('scalar', StandardScaler()), 
        ('poly', PolynomialFeatures(param_dict[MODEL_PARAM_DEG])),
        (param_dict[MODEL_PARAM_MODEL_TYPE], model)
    ])
    return pipe


@api.route('/data/columns', methods=['GET'])
def get_columns():

    if request.method == 'GET':

        datadir = request.json.get(API_KEY_DATADIR)
        data = pd.read_csv(datadir, engine='python')
        data.sort_index(axis=1, inplace=True)
        res = {API_KEY_DATA_COLUMNS: list(data.columns)}

        return res
        

@api.route('/modeling/validation', methods=['POST'])
def model_validation():

    if request.method == 'POST':
        
        param_dict = request.json.get(API_KEY_MODEL_PARAMETER)
        xcols = param_dict[MODEL_PARAM_X_COLUMNS]
        ycol = param_dict[MODEL_PARAM_Y_COLUMN]
        random_state = request.json.get(API_KEY_RANDOM_STATE)

        datadir = request.json.get(API_KEY_DATADIR)
        pipe, grid_param = set_model_validation(param_dict)
        grid = GridSearchCV(pipe, param_grid=grid_param, cv=param_dict[MODEL_PARAM_CV])
        
        dat = pd.read_csv(datadir, engine='python')
        X_train, y_train, X_val, y_val = \
            generate_data(dat, xcols, ycol, random_state=random_state)
        grid.fit(X_train, y_train)

        print(grid.cv_results_)

        res = {
            API_KEY_RESULT_BEST_PARAM: "Best parameters: {}".format(grid.best_params_),
            API_KEY_RESULT_CV_SCORE: "Best cross-validation accuracy: {:.2f}".format(grid.best_score_),
            API_KEY_RESULT_MEAN_FIT_TIME: list(grid.cv_results_[API_KEY_RESULT_MEAN_FIT_TIME]),
            API_KEY_RESULT_MEAN_SCORE_TIME: list(grid.cv_results_[API_KEY_RESULT_MEAN_SCORE_TIME]),
            API_KEY_RESULT_MEAN_TEST_SCORE: list(grid.cv_results_[API_KEY_RESULT_MEAN_TEST_SCORE]),
            API_KEY_RESULT_STD_FIT_TIME: list(grid.cv_results_[API_KEY_RESULT_STD_FIT_TIME]),
            API_KEY_RESULT_STD_TEST_SCORE: list(grid.cv_results_[API_KEY_RESULT_STD_TEST_SCORE]),
            API_KEY_RESULT_STD_SCORE_TIME: list(grid.cv_results_[API_KEY_RESULT_STD_SCORE_TIME]),
            API_KEY_RESULT_VALIDATION_SET_SCORE: "Validation set score: {:.2f}".format(grid.score(X_val, y_val)),
        }

    return res

@api.route('/modeling/test', methods=['GET', 'POST'])
def model_test():

    if request.method == 'GET':
        pass

    elif request.method == 'POST':

        param_dict = request.json.get(API_KEY_MODEL_PARAMETER)
        xcols = param_dict[MODEL_PARAM_X_COLUMNS]
        ycol = param_dict[MODEL_PARAM_Y_COLUMN]
        datadir = request.json.get(API_KEY_DATADIR)
        random_state = request.json.get(API_KEY_RANDOM_STATE)

        X_trainval, y_trainval, X_test, y_test = \
            generate_data(pd.read_csv(datadir, engine='python'), xcols, ycol, phase='test', random_state=random_state)

        pipe = set_model_test(param_dict)

        pipe.fit(X_trainval, y_trainval)
        predict = pipe.predict(X_test)

        score = pipe.score(X_test, y_test)
        mse = mean_squared_error(y_test[ycol[0]].values, predict)
        rmse = np.sqrt(mse)
        rmse_mean = rmse / np.mean(y_test[ycol[0]].values)

        res = {
            API_KEY_RESULT_MSE: mse,
            API_KEY_RESULT_RMSE: rmse,
            API_KEY_RESULT_RMSE_MEAN: rmse_mean,
            API_KEY_RESULT_TEST_SCORE: score
        }

        return res