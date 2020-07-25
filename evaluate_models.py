#!/usr/bin/python

from helper import parse_dot, run_traces_on_model, dict2list, reduce_data_by_label, parse_symbolic_dot, run_traces_on_symbolic_model
from statistics import median
import pandas as pd
import numpy as np
import pickle
import re
import glob
from collections import defaultdict
from operator import add
import os

# Set debugging flag (if set to 0 the interactive version is used)
DEBUGGING = 1


def train_model(traces_filepath, indices_filepath, model, method, clustering_method=None, transformer=None):
    """
    Function for training an input model given some input traces stored in the provided filepath. The traces are firstly
    run on the model, so that each state of the model can be updated with the records passing from it. Subsequently, the
    specified training method is applied on each state, so that a trained model can be created in each state.
    :param traces_filepath: the filepath to the traces' file
    :param indices_filepath: the filepath to the traces' indices limits - used later for prediction
    :param model: the given model
    :param method: the training method to be used (currently probabilistic | multivariate gaussian | clustering | baseline)
    :param clustering_method: the clustering method to be used if clustering has been selected as method, otherwise None
    :param transformer: flag showing if RobustScaler should be used to normalize and scale the data
    :return: the trained model
    """
    model = run_traces_on_model(traces_filepath, indices_filepath, model)
    model.set_all_weights(model.get_maximum_weight(with_laplace=True), with_laplace=True)
    for node_label in model.nodes_dict.keys():
        if node_label != 'root' and len(model.nodes_dict[node_label].observed_indices) > 2:
            if method == 'clustering':
                model.nodes_dict[node_label].training_vars['clusterer'], \
                model.nodes_dict[node_label].training_vars['transformer'] = model.nodes_dict[node_label].\
                    fit_clusters_on_observed(clustering_method, transformer)
            elif method == "multivariate gaussian":
                model.nodes_dict[node_label].training_vars['kernel'], \
                model.nodes_dict[node_label].training_vars['transformer'] = \
                    model.nodes_dict[node_label].fit_multivariate_gaussian(transformer)
            elif method == "probabilistic":
                model.nodes_dict[node_label].training_vars['quantile_values'] = \
                    model.nodes_dict[node_label].fit_quantiles_on_observed()
            else:
                model.nodes_dict[node_label].training_vars['mi'], model.nodes_dict[node_label].training_vars['si'], \
                model.nodes_dict[node_label].training_vars['transformer'] = \
                    model.nodes_dict[node_label].fit_baseline(transformer)
    return model


def predict_on_model(model, method, weighted=True):
    """
    Function for predicting based on a model supplied with the testing traces on its states.
    :param model: the given model
    :param method: the method that has been used for training (needed to select the appropriate prediction mechanism on
    each state)
    :param weighted: a flag indicating if weighted prediction will be applied based on the number of the observations of
    each state (meaning the robustness of the prediction of each state)
    :return: the predicted labels
    """
    predictions = dict()
    weights = dict()
    for node_label in model.nodes_dict.keys():
        # the node needs to have test set to predict on and
        if node_label != 'root' and len(model.nodes_dict[node_label].testing_indices) != 0:
            # TODO: check why the last clause is needed
            if len(model.nodes_dict[node_label].observed_indices) > 2:
                if method == 'clustering':
                    pred = model.nodes_dict[node_label].predict_on_clusters(
                        model.nodes_dict[node_label].training_vars['clusterer'],
                        transformer=model.nodes_dict[node_label].training_vars['transformer'])
                elif method == "multivariate gaussian":
                    pred = model.nodes_dict[node_label].predict_on_gaussian(
                        model.nodes_dict[node_label].training_vars['kernel'],
                        model.nodes_dict[node_label].training_vars['transformer'])
                elif method == "probabilistic":
                    pred = model.nodes_dict[node_label].predict_on_probabilities(
                        model.nodes_dict[node_label].training_vars['quantile_values'])
                else:
                    pred = model.nodes_dict[node_label].predict_on_baseline(
                        model.nodes_dict[node_label].training_vars['mi'],
                        model.nodes_dict[node_label].training_vars['si'],
                        model.nodes_dict[node_label].training_vars['transformer'])
            else:
                # if this state is unseen in training predict anomaly -> this shouldn't happen though
                print('State ' + node_label + ' has less than 3 observations!!!')
                pred = len(model.nodes_dict[node_label].testing_indices) * [1]
            assert (len(pred) == len(model.nodes_dict[node_label].testing_indices)), "Dimension mismatch!!"
            for i, ind in enumerate(model.nodes_dict[node_label].testing_indices):
                if weighted:
                    predictions[ind] = [pred[i] * model.nodes_dict[node_label].weight] if ind not in predictions.keys() \
                        else predictions[ind] + [pred[i] * model.nodes_dict[node_label].weight]
                    weights[ind] = model.nodes_dict[node_label].weight if ind not in weights.keys() \
                        else weights[ind] + model.nodes_dict[node_label].weight
                else:
                    predictions[ind] = [pred[i]] if ind not in predictions.keys() else predictions[ind] + [pred[i]]
    # currently using median to aggregate different predictions for the same flow
    if weighted:
        predictions = dict((k, sum(v) / weights[k]) for k, v in predictions.items())
    else:
        predictions = dict((k, median(v)) for k, v in predictions.items())
    return predictions


def produce_evaluation_metrics(predicted_labels, true_labels, detailed_labels, dst_ips, prediction_type='hard', printing=True):
    """
    Function for calculating the evaluation metrics of the whole pipeline. Depending on the prediction type different
    metrics are calculated. For the hard type the accuracy, the precision, and the recall are provided.
    :param predicted_labels: the predicted labels as a list
    :param true_labels: the true labels as a list
    :param detailed_labels: the detailed labels of the flows as a list
    :param dst_ips: the destination ips of each flow as a list (or None in case of connection level analysis)
    :param prediction_type: the prediction type ("soft" | "hard")
    :param printing: a boolean flag that specifies if the results shall be printed too
    :return: the needed metrics
    """
    if prediction_type == 'hard':
        TP, TN, FP, FN = 0, 0, 0, 0
        # round is applied for rounding in cases of float medians
        predicted_labels = list(map(round, predicted_labels))
        # use 2 dictionaries to keep information about the connections and detailed labels
        detailed_results = dict()
        conn_results = None
        if dst_ips:
            conn_results = dict()
        for i in range(len(true_labels)):
            if detailed_labels[i] not in detailed_results.keys():
                detailed_results[detailed_labels[i]] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            if conn_results is not None and dst_ips[i] not in conn_results.keys():
                conn_results[dst_ips[i]] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            if true_labels[i] == 1:
                if true_labels[i] == predicted_labels[i]:
                    TP += 1
                    detailed_results[detailed_labels[i]]['TP'] += 1
                    if conn_results is not None:
                        conn_results[dst_ips[i]]['TP'] += 1
                else:
                    FN += 1
                    detailed_results[detailed_labels[i]]['FN'] += 1
                    if conn_results is not None:
                        conn_results[dst_ips[i]]['FN'] += 1
            else:
                if true_labels[i] == predicted_labels[i]:
                    TN += 1
                    detailed_results[detailed_labels[i]]['TN'] += 1
                    if conn_results is not None:
                        conn_results[dst_ips[i]]['TN'] += 1
                else:
                    FP += 1
                    detailed_results[detailed_labels[i]]['FP'] += 1
                    if conn_results is not None:
                        conn_results[dst_ips[i]]['FP'] += 1
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = -1 if TP + FP == 0 else TP / (TP + FP)
        recall = -1 if TP + FN == 0 else TP / (TP + FN)
        if printing:
            print('TP: ' + str(TP) + ' TN: ' + str(TN) + ' FP: ' + str(FP) + ' FN:' + str(FN))
            print('Accuracy: ' + str(accuracy))
            print('Precision: ' + str(precision))
            print('Recall: ' + str(recall))
        return TP, TN, FP, FN, accuracy, precision, recall, detailed_results, conn_results
    else:
        # TODO: implement the soft prediction part
        return 0, 0, 0, 0, 0, 0, 0, {}, {}


def print_total_results(results):
    """
    Function for printing the total results aggregated from each connection on each scenario tested
    :param results: the directory with the results for each connection of each testing set produced by each training
    model
    :return:
    """
    for test_set_name in list(filter(lambda x: 'total' in x, results.keys())):
        print('-------------------- Total results for ' + test_set_name + ' --------------------')
        for model_name in results[test_set_name].keys():
            print('---- Model ' + model_name + ' ----')
            if 'symbolic' not in model_name:
                model_TP = results[test_set_name][model_name][0]
                model_TN = results[test_set_name][model_name][1]
                model_FP = results[test_set_name][model_name][2]
                model_FN = results[test_set_name][model_name][3]
                model_accuracy = (model_TP + model_TN) / (model_TP + model_TN + model_FP + model_FN)
                model_precision = -1 if model_TP + model_FP == 0 else model_TP / (model_TP + model_FP)
                model_recall = -1 if model_TP + model_FN == 0 else model_TP / (model_TP + model_FN)
                print('TP: ' + str(model_TP) + ' TN: ' + str(model_TN) + ' FP: ' + str(model_FP) + ' FN:' + str(model_FN))
                print('Accuracy: ' + str(model_accuracy))
                print('Precision: ' + str(model_precision))
                print('Recall: ' + str(model_recall))
            else:
                model_ratio = results[test_set_name][model_name][0]
                print('Ratio: ' + str(model_ratio))


if __name__ == '__main__':
    # set flag regarding the number of flows to be processed TODO: to be removed for final version
    reduction_flag = False
    # set flag for baseline results TODO: to be removed for final version
    baseline_only = False
    # set flag for static window TODO: to be removed for final version
    static = True

    if DEBUGGING:
        # for debugging purposes the following structures can be used
        debug_model_filepaths = sorted(glob.glob(
            'outputs/CTU13/host_level/src_port_dst_port_protocol_num_duration_src_bytes_dst_bytes/scenario3-*_resampled_reduced_static_dfa.dot'),
                                       key=lambda item: re.search('-(.+?)_', item.split('/')[-1]).group(1))
        debug_train_trace_filepaths = sorted(glob.glob(
            'Datasets/CTU13/training/host_level/src_port_dst_port_protocol_num_duration_src_bytes_dst_bytes/scenario3-*-traces_resampled_reduced_static.txt'),
                                             key=lambda item: re.search('-(.+)-', item.split('/')[-1]).group(1))

        debug_methods = [
            # 'clustering'
            # , 'multivariate gaussian'
            # , 'probabilistic'
            'baseline multivariate'
            # 'baseline symbolic'
                         ]

        debug_clustering_methods = [
            'LOF'
            , 'isolation forest'
        ]

        parameters = []
        for model_filepath, trace_filepath in zip(debug_model_filepaths, debug_train_trace_filepaths):
            # check that the right model and trace files are used
            assert (re.search('-(.+?)_', model_filepath.split('/')[-1]).group(1) ==
                    re.search('-(.+)-', trace_filepath.split('/')[-1]).group(1)), "Model-trace mismatch!!"
            for method in debug_methods:
                if method == 'clustering':
                    for clutering_method in debug_clustering_methods:
                        parameters += [(model_filepath, trace_filepath, method, clutering_method)]
                else:
                    parameters += [(model_filepath, trace_filepath, method)]

        flag = 'CTU-bi'
        n = len(parameters)
    else:
        flag = int(input('Provide the type of dataset to be used: '))
        n = int(input('Provide the number of models to be trained: '))
    models = []
    methods = []
    models_info = []
    for i in range(n):
        if DEBUGGING:
            model_filepath = parameters[i][0]
            traces_filepath = parameters[i][1]
            method = parameters[i][2]
        else:
            model_filepath = input('Give the relative path of the model to be used for training: ')
            traces_filepath = input('Give the relative path of the trace to be used for training on the given model: ')
            method = input('Give the name of the training method to be used (clustering | multivariate gaussian | '
                           'probabilistic): ')

        # parse the model from the model dot file
        if method != 'baseline symbolic':
            model = parse_dot(model_filepath)
            indices_filepath = '.'.join(traces_filepath.split('.')[:-1]) + '_indices.pkl'
        else:
            model = parse_symbolic_dot(model_filepath)
            indices_filepath = ''
        clustering_method = None
        if method == 'clustering':
            if DEBUGGING:
                clustering_method = parameters[i][3]
            else:
                clustering_method = input('Provide the specific clustering method to be used (hdbscan | isolation forest '
                                          '| LOF | kmeans): ')
        # train the model
        print('Training on ' + '.'.join(model_filepath.split('/')[-1].split('.')[0:-1]) + '_' + method + '-' +
              (clustering_method if clustering_method is not None else '') + '...')
        # in case the baseline symbolic method is used then there is no need of replaying the training data on the model
        if method != 'baseline symbolic':
            models += [train_model(traces_filepath, indices_filepath, model, method, clustering_method=clustering_method)]
        else:
            models += [model]
        methods += [method + '-' + (clustering_method if clustering_method is not None else '')]
        # list used for better presentation of the results later on
        models_info += ['.'.join(model_filepath.split('/')[-1].split('.')[0:-1]) + '_' + methods[-1]]

    # start testing on each trained model
    if DEBUGGING:
        # get the testing traces filepath pattern
        debug_test_trace_filepaths = sorted(glob.glob(
            'Datasets/CTU13/test/host_level/src_port_dst_port_protocol_num_duration_src_bytes_dst_bytes/scenario13-*-traces_static.txt'))
        # just shortcut for huge inputs used during experimentation - TODO: remove it in final version
        # debug_test_trace_filepaths =sorted(filter(lambda x: '147.32.84.170' not in x, glob.glob('Datasets/CTU13/test/host_level/src_port_dst_port_protocol_num_duration_src_bytes_dst_bytes/scenario3-*-traces_static.txt')))

        debug_test_set_filepaths = list(map(lambda x: '/'.join(x.split('/')[0:2]) + '/'
                                                      + '-'.join(x.split('/')[-1].split('-')[:(-3 if 'connection' in x
                                                                                               else -2)]),
                                            debug_test_trace_filepaths))
        debug_test_filepaths = list(zip(debug_test_trace_filepaths, debug_test_set_filepaths))
        m = len(debug_test_filepaths)
    else:
        m = int(input('Provide the number of testing sets: '))
    results = defaultdict(dict)
    # keep a value showing the last test set tested so that the accumulation of the aggregated results can be refreshed
    prev_test_path = ''
    accumulated_results = defaultdict(list)
    for j in range(m):
        if DEBUGGING:
            test_traces_filepath = debug_test_filepaths[j][0]
        else:
            test_traces_filepath = input('Give the relative path of the testing traces to be used for evaluation: ')
        if 'encoding' not in test_traces_filepath:
            indices_filepath = '.'.join(test_traces_filepath.split('.')[:-1]) + '_indices.pkl'
        else:
            indices_filepath = ''
        # initialize the entry in the results dictionary for the current testing trace file
        test_trace_name = '.'.join(test_traces_filepath.split('/')[-1].split('.')[0:-1])
        print('-------------------------------- Evaluating on ' + test_trace_name + ' --------------------------------')
        results[test_trace_name] = dict()
        # and retrieve the IPs to use for true label extraction
        ips = []
        for ip_tuple in re.findall("-(\d+\.\d+\.\d+\.\d+)|-([^-]+::[^-]+:[^-]+:[^-]+:[^-]+)", test_traces_filepath):
            ips += [ip_tuple[0] if ip_tuple[0] != '' else ip_tuple[1]]
        # retrieve the actual dataset so that the true labels can be extracted
        if DEBUGGING:
            test_data_filepath = debug_test_filepaths[j][1]
        else:
            test_data_filepath = input('Give the relative path of the testing dataframe to be used for evaluation: ')
        if flag == 'CTU-bi':
            normal = pd.read_pickle(test_data_filepath + '/binetflow_normal.pkl')
            anomalous = pd.read_pickle(test_data_filepath + '/binetflow_anomalous.pkl')
        else:
            normal = pd.read_pickle(test_data_filepath + '/normal.pkl')
            anomalous = pd.read_pickle(test_data_filepath + '/anomalous.pkl')
        all_data = pd.concat([normal, anomalous], ignore_index=True).reset_index(drop=True)
        # keep only the flows currently under evaluation based on the ips extracted from the testing traces' filepath
        # and sort values by date
        if len(ips) == 1:
            # host level analysis
            if 'bdr' in test_traces_filepath:
                all_data = all_data[(all_data['src_ip'] == ips[0]) | (all_data['dst_ip'] == ips[0])]\
                    .sort_values(by='date').reset_index(drop=True)
            else:
                all_data = all_data[all_data['src_ip'] == ips[0]].sort_values(by='date').reset_index(drop=True)
        else:
            # connection level analysis
            if 'bdr' in test_traces_filepath:
                all_data = all_data[((all_data['src_ip'] == ips[0]) & (all_data['dst_ip'] == ips[1])) |
                                    ((all_data['dst_ip'] == ips[0]) & (all_data['src_ip'] == ips[1]))] \
                    .sort_values(by='date').reset_index(drop=True)
            else:
                all_data = all_data[(all_data['src_ip'] == ips[0]) & (all_data['dst_ip'] == ips[1])]\
                    .sort_values(by='date').reset_index(drop=True)
        if reduction_flag:
            print('Reducing data points...')
            all_data = reduce_data_by_label(all_data, 10000, flag)  # 7500 for CICIDS | 10000 for UNSW
        true_labels = all_data['label'].values
        # keep also the detailed labels for analysis reasons
        if flag == 'UNSW':
            detailed_labels = all_data['detailed_label'].values.tolist()
        else:
            detailed_labels = all_data['label'].values.tolist()
        # keep also the destination IPs in case we are on host level analysis -> again for analysis reasons
        dst_ips = None
        if len(ips) == 1:
            dst_ips = all_data['dst_ip'].values.tolist()
        # keep one dictionary to aggregate the results of each model over all flows on the test set
        if prev_test_path != test_data_filepath:
            if len(prev_test_path):
                results[prev_test_path + '-total'] = accumulated_results
            accumulated_results = defaultdict(list)
            prev_test_path = test_data_filepath
        for i in range(n):
            print("Let's use model " + models_info[i] + '!!!')
            if methods[i].split('-')[0] != 'baseline symbolic':
                models[i].reset_attributes(attribute_type='test')
                models[i].reset_indices(attribute_type='test')
                models[i] = run_traces_on_model(test_traces_filepath, indices_filepath, models[i], 'test')
                predictions = predict_on_model(models[i], methods[i].split('-')[0])
                assert (len(predictions.keys()) == np.size(true_labels, 0)), \
                    "Dimension mismatch between true and predicted labels!!"
                # Save the results as a dictionary of dictionaries with the first level keys being the test set name,
                # the second level keys being the training model information, and the values being the results
                if flag == 'CTU-bi':
                    results[test_trace_name][models_info[i]] = produce_evaluation_metrics(dict2list(predictions),
                                                                                          list(map(lambda x: 1
                                                                                          if 'Botnet' in x
                                                                                          else 0, true_labels.tolist())),
                                                                                          detailed_labels, dst_ips)
                elif flag == 'UNSW':
                    results[test_trace_name][models_info[i]] = produce_evaluation_metrics(dict2list(predictions),
                                                                                          true_labels.tolist(),
                                                                                          detailed_labels, dst_ips)
                else:
                    results[test_trace_name][models_info[i]] = produce_evaluation_metrics(dict2list(predictions),
                                                                                          list(map(lambda x: 1
                                                                                          if x != 'BENIGN'
                                                                                          else 0, true_labels.tolist())),
                                                                                          detailed_labels, dst_ips)
                # update also the accumulated results | only TP, TN, FP, FN are passed
                if len(accumulated_results[models_info[i]]):
                    accumulated_results[models_info[i]] = list(map(add, accumulated_results[models_info[i]],
                                                                   results[test_trace_name][models_info[i]][0:4]))
                else:
                    accumulated_results[models_info[i]] = list(results[test_trace_name][models_info[i]][0:4])
            else:
                assert (re.search('-(.+?)_', models_info[i].split('/')[-1]).group(1) ==
                        re.search('-(.+)-', parameters[i][1].split('/')[-1]).group(1)), \
                    "Model-trace mismatch in symbolic version!!"
                ratio = run_traces_on_symbolic_model(test_traces_filepath, models[i], eval_method='error',
                                                     train_path=parameters[i][1])  # the training trace filepath

                # in the symbolic case the label of the aggregation entity is given at this point
                if flag == 'CTU-bi':
                    agg_label = any(map(lambda x: 'Botnet' in x, true_labels.tolist()))
                elif flag == 'UNSW':
                    agg_label = any(map(lambda x: x == 1, true_labels.tolist()))
                else:
                    agg_label = any(map(lambda x: x != 'BENIGN', true_labels.tolist()))

                print('Ratio: ' + str(ratio))
                print('Is Malicious: ' + str(agg_label))
                results[test_trace_name][models_info[i]] = ratio, agg_label

                if len(accumulated_results[models_info[i]]):
                    accumulated_results[models_info[i]] = list(map(add, accumulated_results[models_info[i]],
                                                                   results[test_trace_name][models_info[i]][0:1]))
                else:
                    accumulated_results[models_info[i]] = list(results[test_trace_name][models_info[i]][0:1])

    # one last addition of the accumulated results in the results dict
    results[prev_test_path + '-total'] = accumulated_results

    # print the aggregated results for each test set
    print('------------------------- Aggregated results per test set -------------------------')
    print_total_results(results)

    # finally save all the results for each testing trace
    if DEBUGGING:
        results_filename = '/'.join(debug_test_trace_filepaths[0].split('/')[0:2]) + '/' + 'results/' + \
                           debug_test_trace_filepaths[0].split('/')[4] + '/' + \
                           '-'.join(set(map(lambda x: x.split('/')[-1], debug_test_set_filepaths))) + \
                           ('_baseline' if baseline_only else '') + ('_static' if static else '') + '_dfa_results.pkl'
        # create the directory if it does not exist
        os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    else:
        results_filename = input('Provide the relative path for the filename of the results: ')
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
