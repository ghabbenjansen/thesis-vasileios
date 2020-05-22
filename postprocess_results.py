#!/usr/bin/python

import pickle
import glob


def multilevel_statistics(results_dict):
    """
    Function for extracting result statistics on a host and connection level. To label some host or connection according
    to ground truth, a majority voting rule is employed. For predicted values, a 95% confidence value is employed to
    label a host or a connection as benign according to the predictions derived from the bening models on the training
    set.
    :param results_dict: the dictionary with the results obtained for each host by multiple trained models on different
    training methods (LOF, Isolation Forest, Multivariate Gaussian KDE)
    :return: 2 dictionaries with the generated final results on a host and a connection level
    """
    host_results_per_method = {}
    connection_results_per_method = {}
    for host in results_dict.keys():
        # temporary dictionaries for host level analysis
        predicted = {}
        true = {}
        # temporary dictionaries for host connection analysis
        conn_predicted = {}
        conn_true = {}
        # for each training model used
        for result_type in results_dict[host].keys():
            method = result_type.split('_')[-1]
            if method not in predicted.keys():
                predicted[method] = 1
            # The mapping in the dictionary is the following:
            # 0 -> TP, 1 -> TN, 2 -> FP, 3 -> FN,  4 -> accuracy, 5 -> precision, 6 -> recall
            # 7 -> dictionary with type of labels, 8 -> dictionary with the destination IPs
            TP = results_dict[result_type][0]
            TN = results_dict[result_type][1]
            FP = results_dict[result_type][2]
            FN = results_dict[result_type][3]
            benign_ratio = (TN + FN) / (TP + TN + FP + FN)
            # if there is a benign match with more than 95% confidence predict this host as benign
            if benign_ratio > 0.95:
                predicted[method] = 0
            # and assign the real label on the host based on a majority vote
            if TP + FN >= TN + FP:
                true[method] = 1
            else:
                true[method] = 0
            # generate also the results for each connection
            conn_results = results_dict[result_type][-1]
            for dst_ip in conn_results.keys():
                if method not in conn_predicted:
                    conn_predicted[method] = {}
                    conn_true[method] = {}
                if dst_ip not in conn_predicted[method].keys():
                    conn_predicted[method][dst_ip] = 1
                conn_TP = conn_results[dst_ip][0]
                conn_TN = conn_results[dst_ip][1]
                conn_FP = conn_results[dst_ip][2]
                conn_FN = conn_results[dst_ip][3]
                # the same as above is true for the connection level analysis
                conn_benign_ratio = (conn_TN + conn_FN) / (conn_TP + conn_TN + conn_FP + conn_FN)
                if conn_benign_ratio > 0.95:
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
                if true[method] == predicted[method]:
                    host_results_per_method[method]['TP'] += 1
                else:
                    host_results_per_method[method]['FN'] += 1
            else:
                if true[method] == predicted[method]:
                    host_results_per_method[method]['TN'] += 1
                else:
                    host_results_per_method[method]['FP'] += 1
            # then create results for connection level
            if method not in connection_results_per_method.keys():
                connection_results_per_method[method] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
                for dst_ip in conn_true[method].keys():
                    if conn_true[method][dst_ip]:
                        if conn_true[method][dst_ip] == conn_predicted[method][dst_ip]:
                            connection_results_per_method[method]['TP'] += 1
                        else:
                            connection_results_per_method[method]['FN'] += 1
                    else:
                        if conn_true[method][dst_ip] == conn_predicted[method][dst_ip]:
                            connection_results_per_method[method]['TN'] += 1
                        else:
                            connection_results_per_method[method]['FP'] += 1
        return host_results_per_method, connection_results_per_method


if __name__ == '__main__':
    dataset = 'CTU13'
    result_filenames = glob.glob('Datasets/CTU13/scenario*_results.pkl')
    host_level_results = {}
    connection_level_results = {}
    for results_filename in result_filenames:
        scenario = results_filename.split('/')[2].split('_')[0]
        with open(results_filename, 'rb') as f:
            results_dict = pickle.load(f)
        host_level_results[scenario], connection_level_results[scenario] = multilevel_statistics(results_dict)

        print('Host level analysis results for ' + scenario)
        for method in host_level_results[scenario].keys():
            print('--------------- ' + method + ' ---------------')
            print("TP: " + str(host_level_results[scenario][method]['TP']))
            print("TN: " + str(host_level_results[scenario][method]['TP']))
            print("FP: " + str(host_level_results[scenario][method]['TP']))
            print("FN: " + str(host_level_results[scenario][method]['TP']))

        print('Connection level analysis results for ' + scenario)
        for method in connection_level_results[scenario].keys():
            print('--------------- ' + method + ' ---------------')
            print("TP: " + str(connection_level_results[scenario][method]['TP']))
            print("TN: " + str(connection_level_results[scenario][method]['TP']))
            print("FP: " + str(connection_level_results[scenario][method]['TP']))
            print("FN: " + str(connection_level_results[scenario][method]['TP']))
