from helper import parse_dot, run_traces_on_model
from statistics import median


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
                pass    # TODO: define the probabilistic training method
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
                    model.nodes_dict[node_label].training_vars['m'], model.nodes_dict[node_label].training_vars['sigma'])
            else:
                pred = []   # TODO: define the probabilistic testing method

            assert (len(pred) == len(model.nodes_dict[node_label].testing_indices)), "Dimension mismatch!!"
            for i, ind in enumerate(model.nodes_dict[node_label].testing_indices):
                predictions[ind] = [pred[i]] if ind not in predictions.keys() else predictions[ind] + [pred[i]]
    # currently using median to aggregate different predictions for the same flow
    predictions = dict((k, median(v)) for k, v in predictions.items())
    return predictions


if __name__ == '__main__':
    n = int(input('Provide the number of models to be trained: '))
    models = []
    methods = []
    for _ in range(n):
        model_filepath = input('Give the relative path of the model to be used for training: ')
        model = [parse_dot(model_filepath)]
        traces_filepath = input('Give the relative path of the trace to be used for training on the given model: ')
        indices_filepath = traces_filepath.split('.')[0] + '_indices.pkl'
        method = input('Give the name of the training method to be used: ')
        clustering_method = None
        if method == 'clustering':
            clustering_method = input('Provide the specific clustering method to be used: ')
        # for now we train it on Isolation Forest (TODO: train on different training methods)
        models += [train_model(traces_filepath, indices_filepath, model, method, clustering_method=clustering_method)]
        methods += [method + '-' + (clustering_method if clustering_method is not None else '')]

    # start testing on each trained model
    m = int(input('Provide the number of testing sets: '))
    results = []
    for _ in range(m):
        test_traces_filepath = input('Give the relative path of the testing traces to be used for evaluation: ')
        indices_filepath = test_traces_filepath.split('.')[0] + '_indices.pkl'
        for i in range(n):
            models[i].reset_attributes(attribute_type='test')
            models[i].reset_indices(attribute_type='test')
            models[i] = run_traces_on_model(test_traces_filepath, indices_filepath, models[i], 'test')
            predictions = predict_on_model(models[i], methods[i].split('-')[0], methods[i].split('-')[1])
            # TODO: add assertion for the number of flows and if it is equal to the number of predictions
            # TODO: and add the testing part to the true labels
