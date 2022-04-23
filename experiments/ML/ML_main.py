import pandas as pd
import numpy as np
import timeit

from experiments.ML.models.feature_extractor import ts_feature_extraction, ts_features_cleaner
from experiments.ML.models.metrics import aRRMSE, aRMSE, multiRMSE, multiMSE

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, SGDRegressor, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor
from models.MTRegressor import MSVR

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.pipeline import make_pipeline

import shap
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR, LinearSVR

from tsfresh import select_features
import os

from utils.data_utils import read_merged_dataset
from models.model_logger import prediction_plotter_2d, log_run, plot_corr_matrix, plot_true_pred_dist, plot_true_pred_scatter

np.seterr(divide='ignore', invalid='ignore')
os.chdir("../..")
dataset_path = 'data/dataset(s)/24_03_ds1.csv'  # run_history in PyProject folder
dataset_name = dataset_path.split('/')[2].split('.')[0]
seed_list = [26, 22, 111, 5, 1998] # different seed to generate different runs
#seed_list = [26] # different seed to generate different runs

if __name__ == "__main__":
    dataset = read_merged_dataset(dataset_path)
    monitoring_params = ['ParameterSet',
                         'No',
                         # 'Time',
                         'CycleTime',
                         'Heart rate',
                         'Systemic arterial pressure',
                         'Pulmonary arterial pressure',
                         'Right atrial pressure',
                         'Left atrial pressure',
                         'Cardiac output/venous return',
                         'Left ventricular ejection fraction',
                         'Right ventricular ejection fraction',
                         'Hemoglobin',
                         'Systemic arterial oxygen saturation',
                         'Mixed venous oxygen saturation']
    input_params = ['ParameterSet',
                    #'HR',
                    'TotalVascularVolume',
                    'e_lvmax',
                    'e0_lv',
                    'e_rvmax',
                    'e0_rv',
                    'SVR',
                    'PVR',
                    'Ea',
                    'Epa',
                    #'Hb',
                    'O2Cons',
                    'PulmShuntFraction'
                    #'p_low'
                    ]

    # create data and labels
    X_raw = dataset[monitoring_params]
    y = dataset[input_params].groupby(by='ParameterSet').max()

    # check if we already extracted the features, unless, we extract it
    if dataset_name + '-ts_features.csv' in os.listdir('data/dataset(s)/'):
        X = pd.read_csv('data/dataset(s)/'+dataset_name+'-ts_features.csv')
        X = ts_features_cleaner(X)
    else:
        # features extraction #
        # extract all features with ts fresh and save the resulting dataframe
        X = ts_feature_extraction(X_raw, 'data/dataset(s)/'+dataset_name+'-ts_features.csv')

    # Dataset features selection
    #TODO




    # convert data to numpy
    X = X.to_numpy()
    y = y.to_numpy()

    # initialize vector to save scores from different runs
    mse_vector = []
    armsse_vector = []
    multi_mse_vector = []
    multi_rmse_vector = []
    dummy_mse_vector = []
    dummy_arrmse_vector = []
    prediction_vector = []
    runtime_vector = []

    for seed in seed_list:
        np.random.seed(seed)  # seed for reproducibility

        # holdout validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, seed = seed)

        print('Build and fit a regressor model...')
        start = timeit.default_timer()      # start timer to measure computational time

        # choose here the model to use in the MORegressor or RegressorChain
        model = RandomForestRegressor(random_state=seed, n_estimators=100)

        model.fit(X_train, y_train)

        print('##### results run_history (seed: '+str(seed)+')')
        # get model score
        #score = model.score(X_test, y_test)
        #print('Done. Score', score)

        # get model metrics
        y_true = y_test
        y_pred = model.predict(X_test)

        # save y_true and y_pred for further analysis
        prediction_vector.append([y_true, y_pred])

        mse = mean_squared_error(y_true,y_pred)
        print('Done. MSE: ', mse)
        mse_vector.append(mse)

        arrmse = aRRMSE(y_true, y_pred)
        print('Done. aRRMSE: ', arrmse)
        armsse_vector.append(arrmse)

        armse = aRMSE(y_true, y_pred)
        print('Done. aRMSE: ', armse)

        multimse = multiMSE(y_true, y_pred, input_params[1:])
        print('Done. multi MSE: ', multimse)
        multi_mse_vector.append(multimse)

        multirmse = multiRMSE(y_true, y_pred, input_params[1:])
        print('Done. multi RMSE: ', multirmse)
        multi_rmse_vector.append(multirmse)

        stop = timeit.default_timer()
        print('--- runtime: ', stop - start)
        runtime_vector.append(stop - start)

        #plotting

        print('---------Dummy regressor score --------')
        # dummy regressor to compare results
        dummy_regressor = DummyRegressor(strategy='mean')
        dummy_regressor.fit(X_train,y_train)
        y_pred_dummy = dummy_regressor.predict(X_test)

        # get model metrics
        y_true = y_test
        y_pred = dummy_regressor.predict(X_test)
        mse = mean_squared_error(y_true, y_pred)
        print('Done. MSE: ', mse)
        dummy_mse_vector.append(mse)

        arrmse = aRRMSE(y_true, y_pred)
        print('Done. aRRMSE: ', arrmse)
        dummy_arrmse_vector.append(arrmse)
        print('_______________________________________')

    output_dir = 'experiments/ML/run_history/'+"MSVR(kernel = 'linear', gamma = 0.1, epsilon=0.001, C=100)"
    os.mkdir(output_dir)

    # plot distribution (it saves the first run)
    plot_true_pred_dist(prediction_vector,
                        output_dir=output_dir,
                        figname=str(model) + '__' + str(seed_list[0]))

    # save log of the run
    log_run(run_name=str(model),
            output_dir=output_dir,
            mse_vector=mse_vector,
            armsse_vector=armsse_vector,
            multi_mse_vector=multi_mse_vector,
            multi_rmse_vector=multi_rmse_vector,
            dummy_mse_vector=dummy_mse_vector,
            dummy_arrmse_vector=dummy_arrmse_vector,
            prediction_vector=prediction_vector,
            runtime_vector=runtime_vector,
            seed=seed_list)