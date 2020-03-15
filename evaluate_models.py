from helper import parse_dot, run_traces_on_model, dict2list
from statistics import median
import pandas as pd
import numpy as np
import pickle
import re


def train_model(traces_filepath, indices_filepath, model, method, clustering_method=None):
    """
    Function for training an input model given some input traces stored in the provided filepath. The traces are firstly
    run on the model, so that each state of the model can be updated with the records passing from it. Subsequently, the
    specified training method is applied on each state, so that a trained model can be created in each state.
    :param traces_filepath: the filepath to the traces' file
    :param indices_filepath: the filepath to the traces' indices limits - used later for prediction
    :param model: the given model
    :param method: the training method to be used (currently probabilistic | multivariate-gaussian | clustering)
    :param clustering_method: the clustering method to be used if clustering has been selected as method, otherwise None
    :return: the trained model
    """
    model = run_traces_on_model(traces_filepath, indices_filepath, model)
    for node_label in model.nodes_dict.keys():
        if node_label != 'root':
            if method == 'clustering':
                model.nodes_dict[node_label].training_vars['clusterer'] = model.nodes_dict[node_label].\
                    fit_clusters_on_observed(clustering_method)
            elif method == "multivariate-gaussian":
                model.nodes_dict[node_label].training_vars['m'], model.nodes_dict[node_label].training_vars['sigma'] = \
                    model.nodes_dict[node_label].fit_multivariate_gaussian()
            else:
                model.nodes_dict[node_label].training_vars['quantile_values'] = \
                    model.nodes_dict[node_label].fit_quantiles_on_observed()
    return model


def predict_on_model(model, method, clustering_method=''):
    """
    Function for predicting based on a model supplied with the testing traces on its states.
    :param model: the given model
    :param method: the method that has been used for training (needed to select the appropriate prediction mechanism on
    each state)
    :param clustering_method: the clustering method to be used if clustering has been selected as method, otherwise ''
    :return: the predicted labels
    """
    predictions = dict()
    for node_label in model.nodes_dict.keys():
        if node_label != 'root':
            if method == 'clustering':
                pred = model.nodes_dict[node_label].predict_on_clusters(
                    model.nodes_dict[node_label].training_vars['clusterer'], clustering_method=clustering_method)
            elif method == "multivariate-gaussian":
                pred = model.nodes_dict[node_label].predict_on_gaussian(
                    model.nodes_dict[node_label].training_vars['m'],
                    model.nodes_dict[node_label].training_vars['sigma'])
            else:
                pred = model.nodes_dict[node_label].predict_on_probabilities(
                    model.nodes_dict[node_label].training_vars['quantile_values'])

            assert (len(pred) == len(model.nodes_dict[node_label].testing_indices)), "Dimension mismatch!!"
            for i, ind in enumerate(model.nodes_dict[node_label].testing_indices):
                predictions[ind] = [pred[i]] if ind not in predictions.keys() else predictions[ind] + [pred[i]]
    # currently using median to aggregate different predictions for the same flow
    predictions = dict((k, median(v)) for k, v in predictions.items())
    return predictions


def produce_evaluation_metrics(predicted_labels, true_labels, prediction_type='hard', printing=True):
    """
    Function for calculating the evaluation metrics of the whole pipeline. Depending on the prediction type different
    metrics are calculated. For the hard type the accuracy, the precision, and the recall are provided.
    :param predicted_labels: the predicted labels as a list
    :param true_labels: the true labels as a list
    :param prediction_type: the prediction type ("soft" | "hard")
    :param printing: a boolean flag that specifies if the results shall be printed too
    :return: the needed metrics
    """
    if prediction_type == 'hard':
        TP, TN, FP, FN = 0, 0, 0, 0
        for i in range(len(true_labels)):
            if true_labels[i] == 1:
                if true_labels[i] == predicted_labels[i]:
                    TP += 1
                else:
                    FN += 1
            else:
                if true_labels[i] == predicted_labels[i]:
                    TN += 1
                else:
                    FP += 1
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        if printing:
            print('Accuracy: ' + str(accuracy))
            print('Precision: ' + str(precision))
            print('Recall: ' + str(recall))
        return accuracy, precision, recall
    else:
        # TODO: implement the soft prediction part
        return 0, 0, 0


if __name__ == '__main__':
    n = int(input('Provide the number of models to be trained: '))
    models = []
    methods = []
    models_info = []
    for _ in range(n):
        model_filepath = input('Give the relative path of the model to be used for training: ')
        model = [parse_dot(model_filepath)]
        traces_filepath = input('Give the relative path of the trace to be used for training on the given model: ')
        indices_filepath = traces_filepath.split('.')[0] + '_indices.pkl'
        method = input('Give the name of the training method to be used: ')
        clustering_method = None
        if method == 'clustering':
            clustering_method = input('Provide the specific clustering method to be used: ')
        train_data_filepath = input('Give the relative path of the testing dataframe to be used for evaluation: ')
        normal = pd.read_pickle(train_data_filepath + '/zeek_normal.pkl')
        # for now we train it on Isolation Forest (TODO: train on different training methods)
        models += [train_model(traces_filepath, indices_filepath, model, method, clustering_method=clustering_method)]
        methods += [method + '-' + (clustering_method if clustering_method is not None else '')]
        # list used for better presentation of the results later on
        models_info += ['.'.join(model_filepath.split('/')[-1].split('.')[0:-1]) + method]

    # start testing on each trained model - it is assumed that each testing trace corresponds to one host
    m = int(input('Provide the number of testing sets: '))
    results = {}
    for j in range(m):
        test_traces_filepath = input('Give the relative path of the testing traces to be used for evaluation: ')
        indices_filepath = test_traces_filepath.split('.')[0] + '_indices.pkl'
        # initialize the entry in the results dictionary for the current testing trace file
        test_trace_name = '.'.join(test_traces_filepath.split('/')[-1].split('.')[0:-1])
        print('Evaluating on ' + test_trace_name + '...')
        results[test_trace_name] = dict()
        # and retrieve the host IP to use it for true label extraction
        host_ip_matcher = re.search("-(\d+\.\d+\.\d+\.\d+)-|-([^-]+::.+:.+:.+:[^-]+)-", test_traces_filepath)
        host_ip = host_ip_matcher.group(1) if host_ip_matcher.group(1) is not None else host_ip_matcher.group(2)
        # TODO: maybe to be changed for consistency reasons emerging from sorting by date (possbily should be added in
        #  the trace extraction phase
        test_data_filepath = input('Give the relative path of the testing dataframe to be used for evaluation: ')
        normal = pd.read_pickle(test_data_filepath + '/zeek_normal.pkl')
        anomalous = pd.read_pickle(test_data_filepath + '/zeek_anomalous.pkl')
        all_data = pd.concat([normal, anomalous], ignore_index=True).sort_values(by='date')
        true_labels = all_data[all_data['src_ip'] == host_ip]['label'].values
        # needed to map datetimes to indices in case of resampled datasets
        if 'resampled' in test_traces_filepath:
            true_datetimes = all_data[all_data['src_ip'] == host_ip]['date']
        for i in range(n):
            print("Let's use model " + models_info[i] + '!!!')
            models[i].reset_attributes(attribute_type='test')
            models[i].reset_indices(attribute_type='test')
            models[i] = run_traces_on_model(test_traces_filepath, indices_filepath, models[i], 'test')
            predictions = predict_on_model(models[i], methods[i].split('-')[0], methods[i].split('-')[1])
            # TODO: Add function for mapping predictions when resampling has been used - use true datetimes from above
            assert (len(predictions.keys()) == np.size(true_labels, 0)), \
                "Dimension mismatch between true and predicted labels!!"

            # TODO: currently the results are saved as a list of tuples for each testing trace - maybe to be changed to
            #  something more informative - also the mapping between the string representation of the labels and their
            #  int value should become more robust so that other datasets can be used too
            if i == 0:
                results[test_trace_name] = [produce_evaluation_metrics(dict2list(predictions),
                                                                       list(map(lambda x: 1 if x == 'Malicious' else 0,
                                                                                true_labels.tolist())))]
            else:
                results[test_trace_name] += [produce_evaluation_metrics(dict2list(predictions),
                                                                        list(map(lambda x: 1 if x == 'Malicious' else 0,
                                                                                 true_labels.tolist())))]

    # finally save all the results for each testing trace
    results_filename = input('Provide the relative path for the filename of the results: ')
    with open(results_filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
