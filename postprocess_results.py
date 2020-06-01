#!/usr/bin/python

import pickle
import glob
from collections import defaultdict
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt


def generate_thresholds_from_validation(validation_dict, min_host_flows=None, min_conn_flows=None):
    """
    Function for selecting the classification threshold in the case of state machine learning analysis on both the host
    and the connection level. The minimum thresholds needed for classifying correctly each host or connection are
    gathered and plotted as a histogram, so that an fitting threshold to be chosen.
    :param validation_dict: the dictionary with the results obtained for each host by multiple trained models on
    different training methods (LOF, Isolation Forest, Multivariate Gaussian KDE, State Machine baseline). All hosts
    included should be benign, since we are talking about validation data.
    :param min_host_flows: if this flag is set, then only hosts with at least min_host_flows are taken into account in
    the final results
    :param min_conn_flows: if this flag is set, then only connections with at least min_conn_flows are taken into
    account in the final results
    :return: 2 dictionaries with the classification thresholds for each method on each level of analysis
    """
    host_thresholds_per_method = defaultdict(list)
    connection_thresholds_per_method = defaultdict(list)
    for host in validation_dict.keys():
        # for now ignore total results - TODO: change it in the final version of the code
        if 'total' in host:
            continue
        min_thresholds = {}
        min_thresholds_conn = {}
        for result_type in validation_dict[host].keys():
            TP = validation_dict[host][result_type][0]
            TN = validation_dict[host][result_type][1]
            FP = validation_dict[host][result_type][2]
            FN = validation_dict[host][result_type][3]
            # if the min_host_flows flag is set then check if the host has less than min_host_flows
            # in addition check if the host is malicious -> then it should not be included in the validation process
            num_flows = TP + TN + FP + FN
            if (min_host_flows is not None and num_flows < min_host_flows) or TP + FN >= TN + FP:
                continue
            # otherwise the validation process is continued
            method = result_type.split('_')[-1]
            # necessary reformatting for better presentation of some methods
            if method[-1] == '-':
                method = method[:-1]
            if method not in min_thresholds.keys():
                min_thresholds[method] = 1
            anomalous_ratio = (TP + FP) / (TP + TN + FP + FN)
            min_thresholds[method] = min(min_thresholds[method], anomalous_ratio)
            conn_validation_results = validation_dict[host][result_type][-1]
            for dst_ip in conn_validation_results.keys():
                conn_TP = conn_validation_results[dst_ip]['TP']
                conn_TN = conn_validation_results[dst_ip]['TN']
                conn_FP = conn_validation_results[dst_ip]['FP']
                conn_FN = conn_validation_results[dst_ip]['FN']
                # if the min_conn_flows flag is set then check if the connection has less than min_conn_flows
                num_conn_flows = conn_TP + conn_TN + conn_FP + conn_FN
                if min_conn_flows is not None and num_conn_flows < min_conn_flows:
                    continue
                # otherwise the validation process is continued
                if method not in min_thresholds_conn.keys():
                    min_thresholds_conn[method] = {}
                if dst_ip not in min_thresholds_conn[method].keys():
                    min_thresholds_conn[method][dst_ip] = 1
                anomalous_ratio_conn = (conn_TP + conn_FP) / (conn_TP + conn_TN + conn_FP + conn_FN)
                min_thresholds_conn[method][dst_ip] = min(min_thresholds_conn[method][dst_ip], anomalous_ratio_conn)

        for method in min_thresholds.keys():
            host_thresholds_per_method[method].append(1-min_thresholds[method])
            for dst_ip in min_thresholds_conn[method].keys():
                connection_thresholds_per_method[method].append(1-min_thresholds_conn[method][dst_ip])

    # finally decide on the thresholds
    final_host_thresholds = {}
    final_conn_thresholds = {}
    for method in host_thresholds_per_method.keys():
        plt.figure()
        sns.distplot(host_thresholds_per_method[method], bins=10, kde=False)
        plt.title('Threshold distribution for method {} on host level analysis'.format(method))
        plt.xlabel('Threshold (%)')
        plt.ylabel('Occurrence  Count')
        plt.show()
        final_host_thresholds[method] = float(input('Give threshold for ' + method + ' on host level: '))

    for method in connection_thresholds_per_method.keys():
        plt.figure()
        sns.distplot(connection_thresholds_per_method[method], bins=10, kde=False)
        plt.title('Threshold distribution for method {} on connection level analysis'.format(method))
        plt.xlabel('Threshold (%)')
        plt.xlabel('Occurrence  Count')
        plt.show()
        final_conn_thresholds[method] = float(input('Give threshold for ' + method + ' on connection level: '))

    return final_host_thresholds, final_conn_thresholds


def multilevel_statistics(results_dict, host_threshold, connection_threshold, min_host_flows=None, min_conn_flows=None):
    """
    Function for extracting result statistics on a host and connection level. To label some host or connection according
    to ground truth, a majority voting rule is employed. For predicted values, a 95% confidence value is employed to
    label a host or a connection as benign according to the predictions derived from the bening models on the training
    set.
    :param results_dict: the dictionary with the results obtained for each host by multiple trained models on different
    training methods (LOF, Isolation Forest, Multivariate Gaussian KDE, State Machine baseline)
    :param host_threshold: a dictionary with the classification thresholds for host level analysis for each method
    :param connection_threshold: a dictionary with the classification thresholds for connection level analysis for each
    method
    :param min_host_flows: if this flag is set, then only hosts with at least min_host_flows are taken into account in
    the final results
    :param min_conn_flows: if this flag is set, then only connections with at least min_conn_flows are taken into
    account in the final results
    :return: 2 dictionaries with the generated final results on a host and a connection level
    """
    host_results_per_method = {}
    connection_results_per_method = {}
    for host in results_dict.keys():
        # for now ignore total results - TODO: change it in the final version of the code
        if 'total' in host:
            continue
        # temporary dictionaries for host level analysis
        predicted = {}
        true = {}
        # temporary dictionaries for host connection analysis
        conn_predicted = {}
        conn_true = {}
        # for each training model used
        for result_type in results_dict[host].keys():
            # The mapping in the dictionary is the following:
            # 0 -> TP, 1 -> TN, 2 -> FP, 3 -> FN,  4 -> accuracy, 5 -> precision, 6 -> recall
            # 7 -> dictionary with type of labels, 8 -> dictionary with the destination IPs
            TP = results_dict[host][result_type][0]
            TN = results_dict[host][result_type][1]
            FP = results_dict[host][result_type][2]
            FN = results_dict[host][result_type][3]
            # if the min_host_flows flag is set then check if the host has less than min_host_flows
            num_flows = TP + TN + FP + FN
            if min_host_flows is not None and num_flows < min_host_flows:
                continue
            # otherwise the evaluation process is continued
            method = result_type.split('_')[-1]
            # necessary reformatting for better presentation of some methods
            if method[-1] == '-':
                method = method[:-1]
            if method not in predicted.keys():
                predicted[method] = 1
            benign_ratio = (TN + FN) / (TP + TN + FP + FN)
            # if there is a benign match with more than some confidence threshold predict this host as benign
            if benign_ratio > host_threshold[method]:
                predicted[method] = 0
            # and assign the real label on the host based on a majority vote
            if TP + FN >= TN + FP:
                true[method] = 1
            else:
                true[method] = 0
            # generate also the results for each connection
            conn_results = results_dict[host][result_type][-1]
            for dst_ip in conn_results.keys():
                conn_TP = conn_results[dst_ip]['TP']
                conn_TN = conn_results[dst_ip]['TN']
                conn_FP = conn_results[dst_ip]['FP']
                conn_FN = conn_results[dst_ip]['FN']
                # if the num_conn_flows flag is set then check if the connection has less than num_conn_flows
                num_conn_flows = conn_TP + conn_TN + conn_FP + conn_FN
                if min_conn_flows is not None and num_conn_flows < min_conn_flows:
                    continue
                # otherwise the evaluation process is continued
                if method not in conn_predicted:
                    conn_predicted[method] = {}
                    conn_true[method] = {}
                if dst_ip not in conn_predicted[method].keys():
                    conn_predicted[method][dst_ip] = 1
                # the same as above is true for the connection level analysis
                conn_benign_ratio = (conn_TN + conn_FN) / (conn_TP + conn_TN + conn_FP + conn_FN)
                if conn_benign_ratio > connection_threshold[method]:
                    conn_predicted[method][dst_ip] = 0
                if conn_TP + conn_FN >= conn_TN + conn_FP:
                    conn_true[method][dst_ip] = 1
                else:
                    conn_true[method][dst_ip] = 0

        for method in predicted.keys():
            # first create results for host level analysis
            if method not in host_results_per_method.keys():
                host_results_per_method[method] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            if true[method]:
                if predicted[method]:
                    host_results_per_method[method]['TP'] += 1
                else:
                    host_results_per_method[method]['FN'] += 1
            else:
                if predicted[method]:
                    host_results_per_method[method]['FP'] += 1
                else:
                    host_results_per_method[method]['TN'] += 1
            # then create results for connection level
            if method not in connection_results_per_method.keys():
                connection_results_per_method[method] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            for dst_ip in conn_true[method].keys():
                if conn_true[method][dst_ip]:
                    if conn_predicted[method][dst_ip]:
                        connection_results_per_method[method]['TP'] += 1
                    else:
                        connection_results_per_method[method]['FN'] += 1
                else:
                    if conn_predicted[method][dst_ip]:
                        connection_results_per_method[method]['FP'] += 1
                    else:
                        connection_results_per_method[method]['TN'] += 1
    return host_results_per_method, connection_results_per_method


if __name__ == '__main__':
    dataset = 'CTU13'
    methods = [
        'clustering-LOF'
        , 'clustering-isolation forest'
        , 'multivariate gaussian'
        , 'baseline'
    ]
    # first set the classification thresholds using the validation data
    validation_data_filename = 'Datasets/CTU13/scenario3_dfa_results.pkl'
    with open(validation_data_filename, 'rb') as f:
        validation_dict = pickle.load(f)
    # host_threshold, connection_threshold = generate_thresholds_from_validation(validation_dict, 50, 20)
    host_threshold = {}
    connection_threshold = {}
    for method in methods:
        host_threshold[method] = float(input('Give threshold for ' + method + ' on host level: '))
        connection_threshold[method] = float(input('Give threshold for ' + method + ' on connection level: '))
    # then produce the final results for the given testing outputs
    result_filenames = sorted(glob.glob('Datasets/CTU13/scenario*_results.pkl'))
    host_level_results = {}
    connection_level_results = {}
    for results_filename in result_filenames:
        scenario = results_filename.split('/')[2].split('_')[0]
        with open(results_filename, 'rb') as f:
            results_dict = pickle.load(f)
        host_level_results[scenario], connection_level_results[scenario] = multilevel_statistics(results_dict,
                                                                                                 host_threshold,
                                                                                                 connection_threshold,
                                                                                                 50, 5)

        # initialize also the entry for the total results from all scenarios
        if dataset + '_total' not in host_level_results.keys():
            host_level_results[dataset + '_total'] = {}
            connection_level_results[dataset + '_total'] = {}
            for method in methods:
                host_level_results[dataset + '_total'][method] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
                connection_level_results[dataset + '_total'][method] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

        print('===================== Host level analysis results for ' + scenario + ' =====================')
        for method in host_level_results[scenario].keys():
            tp = host_level_results[scenario][method]['TP']
            tn = host_level_results[scenario][method]['TN']
            fp = host_level_results[scenario][method]['FP']
            fn = host_level_results[scenario][method]['FN']
            print('--------------- ' + method + ' ---------------')
            print("TP: {} TN: {} FP: {} FN: {}".format(tp, tn, fp, fn))
            acc = (tp + tn) / (tp + tn + fp + fn)
            prec = -1 if tp + fp == 0 else tp / (tp + fp)
            rec = -1 if tp + fn == 0 else tp / (tp + fn)
            print('Accuracy: ' + str(acc))
            print('Precision: ' + str(prec))
            print('Recall: ' + str(rec))
            host_level_results[dataset + '_total'][method]['TP'] += tp
            host_level_results[dataset + '_total'][method]['TN'] += tn
            host_level_results[dataset + '_total'][method]['FP'] += fp
            host_level_results[dataset + '_total'][method]['FN'] += fn

        print('===================== Connection level analysis results for ' + scenario + ' =====================')
        for method in connection_level_results[scenario].keys():
            tp = connection_level_results[scenario][method]['TP']
            tn = connection_level_results[scenario][method]['TN']
            fp = connection_level_results[scenario][method]['FP']
            fn = connection_level_results[scenario][method]['FN']
            print('--------------- ' + method + ' ---------------')
            print("TP: {} TN: {} FP: {} FN: {}".format(tp, tn, fp, fn))
            acc = (tp + tn) / (tp + tn + fp + fn)
            prec = -1 if tp + fp == 0 else tp / (tp + fp)
            rec = -1 if tp + fn == 0 else tp / (tp + fn)
            print('Accuracy: ' + str(acc))
            print('Precision: ' + str(prec))
            print('Recall: ' + str(rec))
            connection_level_results[dataset + '_total'][method]['TP'] += tp
            connection_level_results[dataset + '_total'][method]['TN'] += tn
            connection_level_results[dataset + '_total'][method]['FP'] += fp
            connection_level_results[dataset + '_total'][method]['FN'] += fn

    print('===================== Host level analysis total results for dataset ' + dataset + ' =====================')
    for method in host_level_results[dataset + '_total'].keys():
        tp = host_level_results[dataset + '_total'][method]['TP']
        tn = host_level_results[dataset + '_total'][method]['TN']
        fp = host_level_results[dataset + '_total'][method]['FP']
        fn = host_level_results[dataset + '_total'][method]['FN']
        print('--------------- ' + method + ' ---------------')
        print("TP: {} TN: {} FP: {} FN: {}".format(tp, tn, fp, fn))
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = -1 if tp + fp == 0 else tp / (tp + fp)
        rec = -1 if tp + fn == 0 else tp / (tp + fn)
        print('Accuracy: ' + str(acc))
        print('Precision: ' + str(prec))
        print('Recall: ' + str(rec))

    print('===================== Connection level analysis results for dataset ' + dataset + ' =====================')
    for method in connection_level_results[dataset + '_total'].keys():
        tp = connection_level_results[dataset + '_total'][method]['TP']
        tn = connection_level_results[dataset + '_total'][method]['TN']
        fp = connection_level_results[dataset + '_total'][method]['FP']
        fn = connection_level_results[dataset + '_total'][method]['FN']
        print('--------------- ' + method + ' ---------------')
        print("TP: {} TN: {} FP: {} FN: {}".format(tp, tn, fp, fn))
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = -1 if tp + fp == 0 else tp / (tp + fp)
        rec = -1 if tp + fn == 0 else tp / (tp + fn)
        print('Accuracy: ' + str(acc))
        print('Precision: ' + str(prec))
        print('Recall: ' + str(rec))
