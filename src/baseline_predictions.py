#!/usr/bin/python

import numpy as np
import pandas as pd
import glob
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from src.helper import select_hosts
sns.set_style("darkgrid")


def create_aggregated_view(data, selected, flag, grouping='host'):
    """
    Function for creating an aggregated view on the data to be used for clustering. The data are aggregated on
    connection level and the label of each connection is assigned according to the existence of anomalous flows.
    :param data: the input dataframe
    :param selected:  the selected features
    :param flag: the flag about the type of the dataset used
    :param grouping: flag regarding the grouping criteria
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
    if grouping == 'host':
        agg_data = data.groupby('src_ip').agg(aggregation_dict).reset_index()
    else:
        agg_data = data.groupby(['src_ip', 'dst_ip']).agg(aggregation_dict).reset_index()
    return agg_data


def evaluate_clustering(data, true, ips, clusterer, printing=True):
    """
    Function for testing LOF on the given data both on a host level.
    :param data: the input data as a numpy array
    :param true: the true labels
    :param ips: the pairs of IP addresses of the dataset
    :param clusterer: the trained LOF clusterer
    :param printing: flag for printing intermediate results
    :return: a list of 2 dictionaries with the results on a host level
    """
    predicted = clusterer.predict(data)
    # change the labels to 0: Benign 1: Malicious
    predicted[predicted == 1] = 0
    predicted[predicted == -1] = 1
    assert (len(predicted) == len(true)), 'Dimension mismatch in true and predicted labels!!!'
    host_results = {}
    for i in range(len(true)):
        # check if we have seen this host before. if not, initialize the entry
        if ips[i] not in host_results.keys():
            host_results[ips[i]] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        if true[i]:
            if predicted[i]:
                host_results[ips[i]]['TP'] += 1
            else:
                host_results[ips[i]]['FN'] += 1
        else:
            if predicted[i]:
                host_results[ips[i]]['FP'] += 1
            else:
                host_results[ips[i]]['TN'] += 1

    # generate the final results - again majority voting is used
    final_results = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for key, res_dict in host_results.items():
        TP = res_dict['TP']
        FP = res_dict['FP']
        TN = res_dict['TN']
        FN = res_dict['FN']
        # if the majority of predictions is anomalous
        if TP + FP > TN + FN:
            # if there is at least one true malicious label
            if TP + FN > 0:
                final_results['TP'] += 1
            # else we have a False positive
            else:
                final_results['FP'] += 1
        # in the opposite case tha handling is similar
        else:
            if TP + FN > 0:
                final_results['FN'] += 1
            else:
                final_results['TN'] += 1
    if printing:
        print('TP: ' + str(final_results['TP']))
        print('TN: ' + str(final_results['TN']))
        print('FP: ' + str(final_results['FP']))
        print('FN: ' + str(final_results['FN']))
    return final_results


if __name__ == '__main__':
    flag = 'CTU-bi'
    training_filepath = 'Datasets/UNSW-NB15/UNSW-NB15-1'
    selected = [
        'src_port'
        , 'dst_port'
        , 'protocol_num'
        # , 'duration'
        , 'src_bytes'
        , 'dst_bytes'
    ]

    # select the desired grouping level
    grouping = 'host'

    # Read the training set consisting only by benign data
    if flag == 'CTU-bi':
        train = pd.read_pickle(training_filepath + '/binetflow_normal.pkl')
    else:
        train = pd.read_pickle(training_filepath + '/normal.pkl')

    # select the major hosts as it is done in the other methods
    train_hosts = list(map(lambda item: item[0], select_hosts(train, 50).values.tolist()))
    train = train[train['src_ip'].isin(train_hosts)]

    # and create an aggregated view on the data so as to cluster them
    agg_train = create_aggregated_view(train, selected, flag, grouping=grouping)
    clusterer = LocalOutlierFactor(novelty=True).fit(agg_train[selected].values)

    # retrieve all the paths to the test sets
    test_filepaths = sorted(glob.glob('Datasets/CTU13/scenario*/'))
    for test_filepath in test_filepaths:
        # just to handle unwanted directories
        if 'results' in test_filepath or 'training' in test_filepath or 'test' in test_filepath:
            continue
        print("=============== Evaluating on " + test_filepath.split('/')[-2] + " =============== ")
        if flag == 'CTU-bi':
            normal = pd.read_pickle(test_filepath + 'binetflow_normal.pkl')
            anomalous = pd.read_pickle(test_filepath + 'binetflow_anomalous.pkl')
        else:
            normal = pd.read_pickle(test_filepath + 'normal.pkl')
            # only of the CICIDS dataset
            if 'Monday' not in test_filepath:
                anomalous = pd.read_pickle(test_filepath + 'anomalous.pkl')
        # only of the CICIDS dataset
        if 'Monday' not in test_filepath:
            all_data = pd.concat([normal, anomalous], ignore_index=True).sort_values(by='date').reset_index(drop=True)
        else:
            all_data = normal

        # select the major hosts as it is done in the other methods
        hosts = list(map(lambda item: item[0], select_hosts(all_data, 50).values.tolist()))
        all_data = all_data[all_data['src_ip'].isin(hosts)]

        # first produce the clustering results
        agg_data = create_aggregated_view(all_data, selected, flag, grouping=grouping)
        true = agg_data['label_num'].values
        if grouping == 'host':
            ips = agg_data['src_ip'].values
        else:
            ips = agg_data[['src_ip', 'dst_ip']].values
        sel_data = agg_data[selected].values
        evaluate_clustering(sel_data, true, ips, clusterer)

