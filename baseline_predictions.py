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
sns.set_style("darkgrid")


def fit_validate_VAR(train_data, validation_data):
    """
    Function for finding the decision thresholds for the VAR models through the use of a validation dataset. Briefly,
    the training set is used to fit the VAR model and a prediction on the validation data is performed. The absolute
    errors per feature are plotted, so that the anomaly threshold for each feature can be selected.
    :param train_data: the training dataframe
    :param validation_data: the validation dataframe
    :return: the fitted model, along with the derived thresholds
    """
    # first fit the model on the training data to evaluate the mse between the fitted model predictions and the
    # evaluation data points
    try:
        model = VAR(endog=train_data).fit(maxlags=20, ic='aic')
    except ValueError:
        print('ValueError occurred -> Current model is skipped')
        return None, None
    except np.linalg.LinAlgError:
        print('LinAlgError occurred -> Current model is skipped')
        return None, None
    pred = model.forecast(model.y, steps=validation_data.shape[0])
    assert (pred.shape[0] == validation_data.shape[0]), "Rows mismatch between predicted data and validation data"
    abs_errors = np.abs(pred - validation_data.values)
    multipliers = []
    for i, col in enumerate(train_data.columns):
        plt.figure()
        sns.distplot(abs_errors[:, i], kde=False)
        plt.title('Absolute Error distribution for ' + col)
        plt.xlabel('Errors')
        plt.ylabel('Occurrence Count')
        plt.show()
        multipliers.append(float(input('Provide the multiplier of the mean absolute error: ')))

    # set the anomaly thresholds as a multiplied value on the MAEs of each attribute
    thresholds = np.multiply(abs_errors.mean(axis=0), np.array(multipliers))
    # then fit the model on the whole dataset
    model = VAR(endog=pd.concat([train_data, validation_data])).fit(maxlags=20, ic='aic')
    # and return both the fitted model and the mse for each feature as an anomaly indicator
    return model, thresholds


def evaluate_VAR(data, models, flag, method='any', printing=False):
    """
    Function for testing VAR on the given data both on a host and a connection level.
    :param data: the input dataframe
    :param models: the fitted VAR models
    :param flag: the flag about the type of the dataset used
    :param method: the method to be used for identifying an anomaly ('any' | 'all' | 'majority')
    :param printing: flag for printing intermediate results
    :return: the final results as a dictionary
    """
    results = defaultdict(dict)
    for src_ip in data['src_ip'].unique():
        host_data = data[data['src_ip'] == src_ip].set_index('date').sort_index()
        dst_ips = host_data['dst_ip'].tolist()
        true = host_data['label'].values
        if flag == 'CTU-bi':
            true = list(map(lambda x: 1 if 'Botnet' in x else 0, true.tolist()))
        elif flag == 'IOT':
            true = list(map(lambda x: 1 if x == 'Malicious' else 0, true.tolist()))
        elif flag == 'UNSW':
            true = true.tolist()
        else:
            true = list(map(lambda x: 1 if x != 'BENIGN' else 0, true.tolist()))
        host_results = {}
        for model_params in models:
            model, thresholds, model_ip = model_params
            pred = model.forecast(model.y, steps=host_data.shape[0])
            assert (pred.shape[0] == host_data.shape[0]), "Rows mismatch between predicted data and evaluation data"
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
            host_results['model_' + model_ip + '_VAR'] = TP, FP, TN, FN, conn_results
        results['results_' + src_ip + '_VAR'] = host_results
    return results


def create_aggregated_view(data, selected, flag):
    """
    Function for creating an aggregated view on the data to be used for clustering. The data are aggregated on
    connection level and the label of each connection is assigned according to the existence of anomalous flows.
    :param data: the input dataframe
    :param selected:  the selected features
    :param flag: the flag about the type of the dataset used
    :return: the aggregated view on the data
    """
    if flag == 'CTU-bi':
        data['label_num'] = data['label'].str.contains('Botnet').astype(np.int)
    elif flag == 'IOT':
        data['label_num'] = data['label'].str.contains('Malicious').astype(np.int)
    elif flag == 'UNSW':
        data['label_num'] = data['label']
    else:
        data['label_num'] = (~data['label'].str.contains('BENIGN')).astype(np.int)

    aggregation_dict = {}
    for feature in selected + ['label_num']:
        if 'label' in feature:
            # by setting max we consider as malicious any connection with at least 1 malicious flow. Other selections
            # would be 'min' if we needed all flows to be malicious or pd.Series.mode if we were considering a majority
            # voting scheme
            aggregation_dict[feature] = 'max'
        elif 'num' in feature:
            aggregation_dict[feature] = lambda x: pd.Series.mode(x)[0]
        else:
            aggregation_dict[feature] = 'mean'

    agg_data = data.groupby(['src_ip', 'dst_ip']).agg(aggregation_dict).reset_index()
    return agg_data


def evaluate_clustering(data, true, ips,  quantile_limit, printing=False):
    """
    Function for testing hdbscan on the given data both on a host and a connection level.
    :param data: the input data as a numpy array
    :param true: the true labels
    :param ips: the pairs of IP addresses of the dataset
    :param quantile_limit: the decision limit for identifying outliers
    :param printing: flag for printing intermediate results
    :return: a list of 2 dictionaries with the results on a host and a connection level
    """
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
        print("=========== Creating VAR model from " + src_ip + " ===========")
        host_data = train[train['src_ip'] == src_ip].set_index('date').sort_index()[selected]
        split_len = int(0.8 * host_data.shape[0])
        train_data, validation_data = host_data.iloc[:split_len-1], host_data.iloc[split_len:]
        VAR_models.append((*fit_validate_VAR(train_data, validation_data), src_ip))
    # filter out skipped models
    VAR_models = list(filter(lambda x: x[0] is not None, VAR_models))

    # then check the outlier scores when using hdbscan clustering on the data
    agg_train = create_aggregated_view(train, selected, flag)
    clusterer = hdbscan.HDBSCAN().fit(agg_train[selected].values)
    plt.figure()
    sns.distplot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], kde=False)
    plt.show()
    quantile_lim = float(input('Give the outliers quantile value according to which a point is considered an outlier: '))

    # retrieve all the paths to the test sets
    test_filepaths = sorted(glob.glob('Datasets/CTU13/scenario*/'))
    for test_filepath in test_filepaths:
        VAR_results = {}
        clustering_results = {}
        if flag == 'CTU-bi':
            normal = pd.read_pickle(test_filepath + 'binetflow_normal.pkl')
            anomalous = pd.read_pickle(test_filepath + 'binetflow_anomalous.pkl')
        else:
            normal = pd.read_pickle(test_filepath + 'normal.pkl')
            anomalous = pd.read_pickle(test_filepath + 'anomalous.pkl')
        all_data = pd.concat([normal, anomalous], ignore_index=True).sort_values(by='date').reset_index(drop=True)

        # first produce the clustering results
        agg_data = create_aggregated_view(all_data, selected, flag)
        true = agg_data['label_num'].values
        ips = agg_data[['src_ip', 'dst_ip']].values
        sel_data = agg_data[selected].values
        filename = '/'.join(test_filepath.split('/')[:-2]) + '/results/' + test_filepath.split('/')[-2] + '_clustering_results.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(evaluate_clustering(sel_data, true, ips, quantile_lim), f, protocol=pickle.HIGHEST_PROTOCOL)

        # then produce the VAR results
        filename = '/'.join(test_filepath.split('/')[:-2]) + '/results/' + test_filepath.split('/')[-2] + '_VAR_results.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(evaluate_VAR(all_data, VAR_models, flag), f, protocol=pickle.HIGHEST_PROTOCOL)
