#!/usr/bin/python

import hdbscan
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from scipy.stats import mode
from collections import defaultdict


def fit_validate_VAR(train_data, validation_data):
    # first fit the model on the training data to evaluate the mse between the fitted model predictions and the
    # evaluation data points
    model = VAR(endog=train_data).fit()
    pred = model.forecast(model.y, steps=validation_data.shape[0])
    abs_errors = np.abs(pred - validation_data.values)
    multipliers = []
    for i, col in enumerate(train_data.columns):
        plt.figure()
        ax = sns.distplot(abs_errors[:, i], kde=False)
        ax.set_title('Absolute Error distribution for ' + col)
        ax.set_xlabel('Errors')
        ax.set_ylabel('Occurrence Count')
        multipliers.append(float(input('Provide the multiplier of the mean absolute error: ')))

    # set the anomaly thresholds as a multiplied value on the MAEs of each attribute
    thresholds = np.multiply(abs_errors.mean(axis=0), np.array(multipliers))
    # then fit the model on the whole dataset
    model = VAR(endog=pd.concat([train_data, validation_data])).fit()
    # and return both the fitted model and the mse for each feature as an anomaly indicator
    return model, thresholds


def evaluate_VAR(data, models, method='any', printing=False):
    results = defaultdict(dict)
    for src_ip in data['src_ip'].unique():
        host_data = data[data['src_ip'] == src_ip].set_index('data').sort_index()
        dst_ips = host_data['dst_ip'].tolist()
        true = host_data['label'].values
        host_results = {}
        for model_params in models:
            model, thresholds, model_ip = model_params
            pred = model.forecast(model.y, steps=host_data.shape[0])
            abs_err = np.abs(pred - host_data[selected].values) > thresholds
            if method == 'any':
                predicted = np.any(abs_err, axis=1).astype(np.int)
            elif method == 'all':
                predicted = np.all(abs_err, axis=1).astype(np.int)
            else:
                predicted = mode(abs_err.astype(np.int), axis=1)[0].flatten()
            assert (len(true) == len(predicted)), 'Dimension mismatch in true and predicted labels!!!'
            TP, FP, TN, FN = 0, 0, 0, 0
            conn_results = {}
            for i in range(len(true)):
                if dst_ips[i] not in conn_results.keys():
                    conn_results[dst_ips[i]] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
                if true[i]:
                    if predicted[i]:
                        TP += 1
                        conn_results[dst_ips[i]]['TP'] += 1
                    else:
                        FN += 1
                        conn_results[dst_ips[i]]['FN'] += 1
                else:
                    if predicted[i]:
                        FP += 1
                        conn_results[dst_ips[i]]['FP'] += 1
                    else:
                        TN += 1
                        conn_results[dst_ips[i]]['TN'] += 1
            if printing:
                print('----------- Results for ' + src_ip + ' from VAR model ' + model_ip + ' -----------')
                print('TP: ' + str(TP))
                print('FP: ' + str(FP))
                print('TN: ' + str(TN))
                print('FN: ' + str(FN))
            host_results['VAR_model_' + model_ip] = TP, FP, TN, FN, conn_results
        results['VAR_results_' + src_ip] = host_results
    return results


def evaluate_clustering(data, true, ips,  quantile_limit, printing=False):
    clusterer = hdbscan.HDBSCAN().fit(data)
    threshold = pd.Series(clusterer.outlier_scores_).quantile(quantile_limit)
    predicted = (clusterer.outlier_scores_ > threshold).astype(np.int)
    assert (len(predicted) == len(true)), 'Dimension mismatch in true and predicted labels!!!'
    host_results = {}
    conn_results = {}
    for i in range(len(true)):
        # check if we have seen this host before. if not, initialize the entry
        if ips[i][0] not in host_results.keys():
            host_results[ips[i][0]] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        # then check if we have seen this connection before. if not, initialize the entry
        if ips[i][0] + '-' + ips[i][1] not in conn_results.keys():
            conn_results[ips[i][0] + '-' + ips[i][1]] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        if true[i]:
            if predicted[i]:
                host_results[ips[i][0]]['TP'] += 1
                conn_results[ips[i][0] + '-' + ips[i][1]]['TP'] += 1
            else:
                host_results[ips[i][0]]['FN'] += 1
                conn_results[ips[i][0] + '-' + ips[i][1]]['FN'] += 1
        else:
            if predicted[i]:
                host_results[ips[i][0]]['FP'] += 1
                conn_results[ips[i][0] + '-' + ips[i][1]]['FP'] += 1
            else:
                host_results[ips[i][0]]['TN'] += 1
                conn_results[ips[i][0] + '-' + ips[i][1]]['TN'] += 1

    # generate the final results - again majority voting is used
    final_results = [{'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}, {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}]
    for i, results in enumerate([host_results, conn_results]):
        for key, res_dict in results.items():
            TP = res_dict['TP']
            FP = res_dict['FP']
            TN = res_dict['TN']
            FN = res_dict['FN']
            if printing:
                print('-------------- Evaluation results for ' + key + ' -------------- ')
                print('TP: ' + str(TP))
                print('FP: ' + str(FP))
                print('TN: ' + str(TN))
                print('FN: ' + str(FN))
            # if the majority of predictions is anomalous
            if TP + FP > TN + FN:
                # if also the majority of the true labels is anomalous, then we have a True positive
                if TP + FN > TN + FP:
                    final_results[i]['TP'] += 1
                # else we have a False positive
                else:
                    final_results[i]['FP'] += 1
            # in the opposite case tha handling is similar
            else:
                if TP + FN > TN + FP:
                    final_results[i]['FN'] += 1
                else:
                    final_results[i]['TN'] += 1
    return final_results


if __name__ == '__main__':
    flag = 'CTU-bi'
    training_filepath = 'Datasets/CTU13/scenario3'
    selected = [
        # 'src_port'
        'dst_port'
        , 'protocol_num'
        # , 'duration'
        , 'src_bytes'
        , 'dst_bytes'
    ]

    # Read the training set consisting only by benign data
    if flag == 'CTU-bi':
        train = pd.read_pickle(training_filepath + '/binetflow_normal.pkl')
    else:
        train = pd.read_pickle(training_filepath + '/normal.pkl')

    # first learn a VAR model on each host and store as anomaly indicators the mse of each feature
    VAR_models = []
    for src_ip in train['src_ip'].unique().tolist():
        host_data = train[train['src_ip'] == src_ip].set_index('date').sort_index()[selected]
        split_len = int(0.8 * host_data.shape[0])
        train_data, validation_data = host_data.iloc[:split_len], host_data.iloc[split_len:]
        VAR_models.append((*fit_validate_VAR(train_data, validation_data), src_ip))

    # then check the outlier scores when using hdbscan clustering on the data
    clusterer = hdbscan.HDBSCAN().fit(train[selected].values)
    plt.figure()
    sns.distplot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], kde=False)
    quantile_lim = float(input('Give the outliers quantile value according to which a point is considered an outlier: '))

    # retrieve all the paths to the test sets
    test_filepaths = glob.glob('Datasets/CTU13/scenario*/')
    for test_filepath in test_filepaths:
        VAR_results = {}
        clustering_results = {}
        if flag == 'CTU-bi':
            normal = pd.read_pickle(test_filepath + '/binetflow_normal.pkl')
            anomalous = pd.read_pickle(test_filepath + '/binetflow_anomalous.pkl')
        else:
            normal = pd.read_pickle(test_filepath + '/normal.pkl')
            anomalous = pd.read_pickle(test_filepath + '/anomalous.pkl')
        all_data = pd.concat([normal, anomalous], ignore_index=True).sort_values(by='date').reset_index(drop=True)

        # first produce the clustering results
        sel_data = all_data[selected].values
        true = all_data['label'].values
        ips = all_data[['src_ip', 'dst_ip']].values
        filename = '/'.join(test_filepath.split('/')[:-1]) + test_filepath.split('/')[-1] + '_clustering_results.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(evaluate_clustering(sel_data, true, ips, quantile_lim), f, protocol=pickle.HIGHEST_PROTOCOL)

        # then produce the VAR results
        filename = '/'.join(test_filepath.split('/')[:-1]) + test_filepath.split('/')[-1] + '_VAR_results.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(evaluate_VAR(all_data, VAR_models), f, protocol=pickle.HIGHEST_PROTOCOL)
