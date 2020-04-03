from helper import parse_dot, run_traces_on_model, dict2list
from statistics import median
import pandas as pd
import numpy as np
import pickle
import re
import glob

debugging = 1


def train_model(traces_filepath, indices_filepath, model, method, clustering_method=None):
    """
    Function for CTU13 an input model given some input traces stored in the provided filepath. The traces are firstly
    run on the model, so that each state of the model can be updated with the records passing from it. Subsequently, the
    specified CTU13 method is applied on each state, so that a trained model can be created in each state.
    :param traces_filepath: the filepath to the traces' file
    :param indices_filepath: the filepath to the traces' indices limits - used later for prediction
    :param model: the given model
    :param method: the CTU13 method to be used (currently probabilistic | multivariate gaussian | clustering)
    :param clustering_method: the clustering method to be used if clustering has been selected as method, otherwise None
    :return: the trained model
    """
    model = run_traces_on_model(traces_filepath, indices_filepath, model)
    model.set_all_weights(model.get_maximum_weight())
    for node_label in model.nodes_dict.keys():
        # TODO: understand why the second clause is needed -> shouldn't be
        if node_label != 'root' and len(model.nodes_dict[node_label].observed_indices) > 2:
            if method == 'clustering':
                model.nodes_dict[node_label].training_vars['clusterer'], \
                model.nodes_dict[node_label].training_vars['transformer'] = model.nodes_dict[node_label].\
                    fit_clusters_on_observed(clustering_method)
            elif method == "multivariate gaussian":
                # model.nodes_dict[node_label].training_vars['m'], model.nodes_dict[node_label].training_vars['sigma'] = \
                #     model.nodes_dict[node_label].fit_multivariate_gaussian()
                model.nodes_dict[node_label].training_vars['kernel'], \
                model.nodes_dict[node_label].training_vars['transformer'] = \
                    model.nodes_dict[node_label].fit_multivariate_gaussian()
            else:
                model.nodes_dict[node_label].training_vars['quantile_values'] = \
                    model.nodes_dict[node_label].fit_quantiles_on_observed()
    return model


def predict_on_model(model, method, clustering_method='', weighted=True):
    """
    Function for predicting based on a model supplied with the testing traces on its states.
    :param model: the given model
    :param method: the method that has been used for CTU13 (needed to select the appropriate prediction mechanism on
    each state)
    :param clustering_method: the clustering method to be used if clustering has been selected as method, otherwise ''
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
                        model.nodes_dict[node_label].training_vars['clusterer'], clustering_method=clustering_method,
                        transformer=model.nodes_dict[node_label].training_vars['transformer'])
                elif method == "multivariate gaussian":
                    # pred = model.nodes_dict[node_label].predict_on_gaussian(
                    #     model.nodes_dict[node_label].training_vars['m'],
                    #     model.nodes_dict[node_label].training_vars['sigma'])
                    pred = model.nodes_dict[node_label].predict_on_gaussian(
                        model.nodes_dict[node_label].training_vars['kernel'],
                        model.nodes_dict[node_label].training_vars['transformer'])
                else:
                    pred = model.nodes_dict[node_label].predict_on_probabilities(
                        model.nodes_dict[node_label].training_vars['quantile_values'])
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


def dates2indices(date_dict, dates):
    """
    Function for propagating the values of a dictionary with resampled date indices as its keys to the actual indices
    before resampling using a series object containing the actual dates
    :param date_dict: a dictionary with keys the resampled dates
    :param dates: a Series of the actual dates
    :return: a dictionary with the indices of the actual dates as keys and the propagated values as its values
    """
    # first keep the resampled dates in a dataframe and sort them
    date_df = pd.DataFrame({'resampled_dates': list(date_dict.keys())}).sort_values(by='resampled_dates')
    ind = 0
    new_dict = dict()
    for items in dates.iteritems():
        # if the current resampled date examined is the last one then just check for lower bound for the actual dates
        if ind == date_df.shape[0] - 1:
            if date_df.resampled_dates[ind] <= items[1]:
                new_dict[items[0]] = date_dict[date_df.resampled_dates[ind]]
            else:
                print("This clause should not be accessed -> Error !!!!!!!!!!!")
        # otherwise check both upper and lower limits
        else:
            # the first clause applies only for flows in the beginning of the recording that weren't captured by the
            # resampler. In this case our predictions are biased towards the benign class
            if items[1] < date_df.resampled_dates[ind]:
                new_dict[items[0]] = 0
            # the second clause captures all flows belonging between two resampled timestamps. In this case the
            # predicted label of the lower resampled timestamp is assigned to these flows
            elif date_df.resampled_dates[ind] <= items[1] < date_df.resampled_dates[ind+1]:
                new_dict[items[0]] = date_dict[date_df.resampled_dates[ind]]
            # the third clause captures the flow that forces into a change of the resampled window that we are currently
            # examining
            else:
                # and if the upper limit is violated increment the resampled dates' index
                ind += 1
                new_dict[items[0]] = date_dict[date_df.resampled_dates[ind]]
    return new_dict


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
        # floor is applied for rounding in cases of float medians
        predicted_labels = list(map(round, predicted_labels))
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
        precision = -1 if TP + FP == 0 else TP / (TP + FP)
        recall = -1 if TP + FN == 0 else TP / (TP + FN)
        if printing:
            print('TP: ' + str(TP) + ' TN: ' + str(TN) + ' FP: ' + str(FP) + ' FN:' + str(FN))
            print('Accuracy: ' + str(accuracy))
            print('Precision: ' + str(precision))
            print('Recall: ' + str(recall))
        return accuracy, precision, recall
    else:
        # TODO: implement the soft prediction part
        return 0, 0, 0


if __name__ == '__main__':
    if debugging:
        # for debugging purposes the following structures can be used
        debug_model_filepaths = glob.glob('outputs/IOT23/connection_level/protocol_num_orig_ip_bytes_resp_ip_bytes/Malware-Capture-1-1*.dot')
        debug_train_trace_filepaths = glob.glob('Datasets/IOT23/training/connection_level/protocol_num_orig_ip_bytes_resp_ip_bytes/Malware-Capture-1-1*.txt')

        debug_methods = [
            'clustering'
            # , 'multivariate gaussian'
            # , 'probabilistic'
                         ]

        debug_clustering_methods = [
            # 'hdbscan'
            # ,
            'LOF'
            , 'isolation forest'
            # , 'kmeans'
        ]

        parameters = []
        for model_filepath, trace_filepath in zip(debug_model_filepaths, debug_train_trace_filepaths):
            for method in debug_methods:
                if method == 'clustering':
                    for clutering_method in debug_clustering_methods:
                        parameters += [(model_filepath, trace_filepath, method, clutering_method)]
                else:
                    parameters += [(model_filepath, trace_filepath, method)]

        flag = 'IOT'
        n = len(parameters)
    else:
        flag = int(input('Provide the type of dataset to be used: '))
        n = int(input('Provide the number of models to be trained: '))
    models = []
    methods = []
    models_info = []
    for i in range(n):
        if debugging:
            model_filepath = parameters[i][0]
        else:
            model_filepath = input('Give the relative path of the model to be used for training: ')
        model = parse_dot(model_filepath)
        if debugging:
            traces_filepath = parameters[i][1]
        else:
            traces_filepath = input('Give the relative path of the trace to be used for training on the given model: ')
        indices_filepath = '.'.join(traces_filepath.split('.')[:-1]) + '_indices.pkl'
        if debugging:
            method = parameters[i][2]
        else:
            method = input('Give the name of the training method to be used (clustering | multivariate gaussian | '
                           'probabilistic): ')
        clustering_method = None
        if method == 'clustering':
            if debugging:
                clustering_method = parameters[i][3]
            else:
                clustering_method = input('Provide the specific clustering method to be used (hdbscan | isolation forest '
                                          '| LOF | kmeans): ')
        # train the model
        print('Training on ' + '.'.join(model_filepath.split('/')[-1].split('.')[0:-1]) + '_' + method + '-' + (clustering_method if clustering_method is not None else '') + '...')
        models += [train_model(traces_filepath, indices_filepath, model, method, clustering_method=clustering_method)]
        methods += [method + '-' + (clustering_method if clustering_method is not None else '')]
        # list used for better presentation of the results later on
        models_info += ['.'.join(model_filepath.split('/')[-1].split('.')[0:-1]) + '_' + methods[-1]]

    # start testing on each trained model - it is assumed that each testing trace corresponds to one host
    if debugging:
        debug_test_trace_filepaths = glob.glob('Datasets/IOT23/test/connection_level/protocol_num_orig_ip_bytes_resp_ip_bytes/Malware-Capture-20-1*.txt')
        debug_test_filepaths = list(zip(debug_test_trace_filepaths, len(debug_test_trace_filepaths) * ['Datasets/IOT23/Malware-Capture-20-1']))
        m = len(debug_test_filepaths)
    else:
        m = int(input('Provide the number of testing sets: '))
    results = {}
    for j in range(m):
        if debugging:
            test_traces_filepath = debug_test_filepaths[j][0]
        else:
            test_traces_filepath = input('Give the relative path of the testing traces to be used for evaluation: ')
        indices_filepath = '.'.join(test_traces_filepath.split('.')[:-1]) + '_indices.pkl'
        # initialize the entry in the results dictionary for the current testing trace file
        test_trace_name = '.'.join(test_traces_filepath.split('/')[-1].split('.')[0:-1])
        print('-------------------------------- Evaluating on ' + test_trace_name + ' --------------------------------')
        results[test_trace_name] = dict()
        # and retrieve the IPs to use for true label extraction
        ips = []
        for ip_tuple in re.findall("-(\d+\.\d+\.\d+\.\d+)|-([^-]+::[^-]+:[^-]+:[^-]+:[^-]+)", test_traces_filepath):
            ips += [ip_tuple[0] if ip_tuple[0] != '' else ip_tuple[1]]
        # retrieve the actual dataset so that the true labels can be extracted
        if debugging:
            test_data_filepath = debug_test_filepaths[j][1]
        else:
            test_data_filepath = input('Give the relative path of the testing dataframe to be used for evaluation: ')
        if flag == 'CTU-bi':
            normal = pd.read_pickle(test_data_filepath + '/binetflow_normal.pkl')
            anomalous = pd.read_pickle(test_data_filepath + '/binetflow_anomalous.pkl')
        elif flag == 'IOT':
            normal = pd.read_pickle(test_data_filepath + '/zeek_normal.pkl')
            anomalous = pd.read_pickle(test_data_filepath + '/zeek_anomalous.pkl')
        else:
            normal = pd.read_pickle(test_data_filepath + '/normal.pkl')
            anomalous = pd.read_pickle(test_data_filepath + '/anomalous.pkl')
        all_data = pd.concat([normal, anomalous], ignore_index=True).reset_index(drop=True)
        # keep only the flows currently under evaluation based on the ips extracted from the testing traces' filepath
        # and sort values by date
        if len(ips) == 1:
            # host level analysis
            all_data = all_data[all_data['src_ip'] == ips[0]].sort_values(by='date').reset_index(drop=True)
        else:
            # connection level analysis
            all_data = all_data[(all_data['src_ip'] == ips[0]) & (all_data['dst_ip'] == ips[1])].sort_values(by='date')\
                .reset_index(drop=True)
        true_labels = all_data['label'].values
        # needed to map datetimes to indices in case of resampled datasets
        true_datetimes = all_data['date'] if 'resampled' in test_traces_filepath else None
        for i in range(n):
            print("Let's use model " + models_info[i] + '!!!')
            models[i].reset_attributes(attribute_type='test')
            models[i].reset_indices(attribute_type='test')
            models[i] = run_traces_on_model(test_traces_filepath, indices_filepath, models[i], 'test')
            predictions = predict_on_model(models[i], methods[i].split('-')[0], methods[i].split('-')[1])
            if true_datetimes is not None:
                predictions = dates2indices(predictions, true_datetimes)
            assert (len(predictions.keys()) == np.size(true_labels, 0)), \
                "Dimension mismatch between true and predicted labels!!"

            # Save the results as a dictionary of dictionaries with the first level keys being the test set name, the
            # second level keys being the tre CTU13 model information, and the values being the results
            if flag == 'CTU-bi':
                if i == 0:
                    results[test_trace_name] = {models_info[i]: produce_evaluation_metrics(dict2list(predictions),
                                                                                           list(map(lambda x: 1
                                                                                           if 'Botnet' in x else 0,
                                                                                                    true_labels.tolist())))}
                else:
                    results[test_trace_name][models_info[i]] = produce_evaluation_metrics(dict2list(predictions),
                                                                                          list(map(lambda x: 1
                                                                                          if 'Botnet' in x else 0,
                                                                                                   true_labels.tolist())))
            elif flag == 'IOT':
                if i == 0:
                    results[test_trace_name] = {models_info[i]: produce_evaluation_metrics(dict2list(predictions),
                                                                                           list(map(lambda x: 1
                                                                                           if x == 'Malicious' else 0,
                                                                                                    true_labels.tolist())))}
                else:
                    results[test_trace_name][models_info[i]] = produce_evaluation_metrics(dict2list(predictions),
                                                                                          list(map(lambda x: 1
                                                                                          if x == 'Malicious' else 0,
                                                                                                   true_labels.tolist())))
            elif flag == 'UNSW':
                if i == 0:
                    results[test_trace_name] = {models_info[i]: produce_evaluation_metrics(dict2list(predictions),
                                                                                           true_labels.tolist())}
                else:
                    results[test_trace_name][models_info[i]] = produce_evaluation_metrics(dict2list(predictions),
                                                                                          true_labels.tolist())
            else:
                if i == 0:
                    results[test_trace_name] = {models_info[i]: produce_evaluation_metrics(dict2list(predictions),
                                                                                           list(map(lambda x: 1
                                                                                           if x != 'BENIGN' else 0,
                                                                                                    true_labels.tolist())))}
                else:
                    results[test_trace_name][models_info[i]] = produce_evaluation_metrics(dict2list(predictions),
                                                                                          list(map(lambda x: 1
                                                                                          if x != 'BENIGN' else 0,
                                                                                                   true_labels.tolist())))

    # finally save all the results for each testing trace
    results_filename = input('Provide the relative path for the filename of the results: ')
    with open(results_filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
