import operator
import numpy as np
import pandas as pd
import hdbscan
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from math import pi, sqrt, ceil


class ModelNode:
    inequality_mapping = {
        '<': operator.lt,
        '>': operator.gt,
        '<=': operator.le,
        '>=': operator.ge,
        '==': operator.eq
    }

    def __init__(self, label, attributes, fin, total, dst_nodes, tran_conditions):
        """
        Init function for the ModelNode class used to assign values to some necessary variables when a node is created
        :param label: the label of the node (String)
        :param attributes: the quantile information of each attribute (Dictionary)
        :param fin: the number of times this node/state was final during the creation of the model (int)
        :param total: the number of times this node/state was visited during the creation of the model (int)
        :param dst_nodes: the nodes to which there are transitions from the current node (List)
        :param tran_conditions: the inequalities of each transition (Dictionary)
        """
        self.label = label
        self.attributes = attributes
        self.fin = fin
        self.total = total
        self.dst_nodes = dst_nodes
        self.tran_conditions = tran_conditions
        # observed attributes is used to store the values observed for each attribute when a trace file is run on the
        # model
        self.observed_attributes = dict(zip(self.attributes.keys(), len(self.attributes.keys()) * [[]]))
        self.observed_indices = []
        self.testing_attributes = dict(zip(self.attributes.keys(), len(self.attributes.keys()) * [[]]))
        self.testing_indices = []
        # dictionary for storing any variables needed for training
        self.training_vars = dict()
        # dictionary for the probabilities associated with each quantile
        # It is initialized to the number of observations in each quantile divided by the total number of observations
        # for the examined attribute. 1 was added to each number so that 0 probabilities will be avoided.
        self.quantile_probs = dict(zip(self.attributes.keys(), (((np.array(self.attributes[attribute]) + 1)
                                                                / (sum(self.attributes[attribute]) + 4)).tolist()
                                   for attribute in self.attributes.keys())))
        # the importance weight of each node - used for weighted predictions
        self.weight = 0

    def evaluate_transition(self, dst_node_label, input_attributes):
        """
        Function for evaluating if a transition from a source to a destination state under the provided input attributes'
        values can be fired
        :param dst_node_label: the label of the destination state
        :param input_attributes: the input attributes' values
        :return: a boolean value denoting the ability to fire the transition
        """
        return all([self.inequality_mapping[condition[1]](input_attributes[condition[0]], condition[2])
                    for condition in self.tran_conditions[dst_node_label]])

    def reset_observed_attributes(self):
        """
        Function for resetting the observed attributes of the node
        :return:
        """
        self.observed_attributes = dict(zip(self.attributes.keys(), len(self.attributes.keys()) * [[]]))

    def reset_observed_indices(self):
        """
        Function for resetting the observed indices of the node
        :return:
        """
        self.observed_indices = []

    def reset_testing_attributes(self):
        """
        Function for resetting the testing attributes of the node
        :return:
        """
        self.testing_attributes = dict(zip(self.attributes.keys(), len(self.attributes.keys()) * [[]]))

    def reset_testing_indices(self):
        """
        Function for resetting the testing indices of the node
        :return:
        """
        self.testing_indices = []

    def attributes2dataset(self, input_attributes):
        """
        Function for converting some given attributes dict to a dataframe to be processed
        like a dataset (m examples x n attributes)
        :param input_attributes: the input attributes' values
        :return: the dataset as a dataframe
        """
        return pd.DataFrame.from_dict(input_attributes)

    def fit_quantiles_on_observed(self):
        """
        Function for calculating the values associated with each of the 3 main quantiles (25%, 50%, 75%) for each
        attribute. If there are no observed attributes on the node, an empty dictionary is returned instead.
        :return: a dictionary with the attributes as a key and the quantile values as a 3-element list
        """
        if bool(self.observed_indices):
            quantile_values = dict()
            for attribute, attribute_values in self.observed_attributes.items():
                quantile_values[attribute] = [np.quantile(np.array(attribute_values), 0.25)]
                quantile_values[attribute] += [np.quantile(np.array(attribute_values), 0.5)]
                quantile_values[attribute] += [np.quantile(np.array(attribute_values), 0.75)]
            return quantile_values
        else:
            print('No observed attributes on node ' + self.label)
            return dict()

    def predict_on_probabilities(self, quantile_values, epsilon=0.0001, prediction_type='hard'):
        """
        Function for predicting anomalies given the quantile probabilities of a node. Given the observed values for each
        attribute and the quantile in which they belong, the associated probabilities are retrieved and the anomaly
        score is calculated as the product of the probabilities of all attributes for each flow.
        :param quantile_values: the quantile limits for each attribute as a dictionary with the keys being the
        attributes and with values the 3-element list generated by the fit_quantiles_on_observed function
        :param epsilon: the detection threshold
        :param prediction_type: the prediction type (hard or soft)
        :return: the prediction labels
        """
        x_test = self.attributes2dataset(self.testing_attributes).values
        n_cols = np.size(x_test, 1)     # retrieve the number of columns, meaning the number of attributes
        # create a vectorized function for finding the quantile index of an attribute value given the quantile limits
        vectorized_quantile_num = np.vectorize(lambda x, y_list: len([y for y in y_list if x > y]), excluded=['y_list'])
        # apply the vectorized function for each attribute (column) and create a new array with the quantile indices
        # as its content and with the same dimensions as the input numpy array
        x_test_quantiles = np.column_stack(tuple([vectorized_quantile_num(x=x_test[:, i],
                                                                          y_list=quantile_values[str(i)])
                                                  for i in range(n_cols)]))
        # create a vectorized function for finding the probability of each attribute value given its quantile index
        vectorized_quantile_probs = np.vectorize(lambda x, y_list: y_list[x], excluded=['y_list'])
        # apply the vectorized function for each attribute (column) and create a new array with the same dimensions as
        # the input numpy array containing the probabilities for each attribute value
        x_test_probs = np.column_stack(tuple([vectorized_quantile_probs(x=x_test_quantiles[:, i],
                                                                        y_list=self.quantile_probs[str(i)])
                                              for i in range(n_cols)]))
        # and finally produce the anomaly score for each flow by multiplying the probabilities of its attribute values
        test_labels = np.prod(x_test_probs, axis=1)
        if prediction_type == 'hard':
            test_labels = (test_labels < epsilon).astype(np.int)
        return test_labels

    def fit_clusters_on_observed(self, clustering_method='kmeans'):
        """
        Function for fitting clusters on the data points observed at each state/node
        :param clustering_method: the clustering method (currently kmeans | hdbscan | Isolation Forest | LOF)
        :return: the fitted cluster estimator, and a normalization transformer in case it was used
        """
        x_train = self.attributes2dataset(self.observed_attributes).values
        transformer = None
        if clustering_method == "hdbscan":
            transformer = RobustScaler().fit(x_train)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=ceil(x_train.shape[0]/2), allow_single_cluster=True,
                                        prediction_data=True).fit(transformer.transform(x_train))
        elif clustering_method == "isolation forest":
            clusterer = IsolationForest().fit(x_train)
        elif clustering_method == "LOF":
            clusterer = LocalOutlierFactor(n_neighbors=ceil(x_train.shape[0]/10), novelty=True).fit(x_train)
        else:
            clusterer = KMeans(n_clusters=2).fit(x_train)
        return clusterer, transformer

    def predict_on_clusters(self, clusterer, clustering_method='kmeans', clustering_type='hard', transformer=None):
        """
        Function for predicting the cluster labels on the testing traces of the node given a fitted cluster estimator
        :param clusterer: the fitted cluster estimator
        :param clustering_method: the clustering method (currently kmeans | hdbscan | Isolation Forest | LOF)
        :param clustering_type: the clustering type to be used (hard or soft)
        :param transformer: the normalization transformer in case one was used
        :return: the predicted labels
        """
        x_test = self.attributes2dataset(self.testing_attributes).values
        if transformer is not None:
            x_test = transformer.transform(x_test)
        if clustering_type == 'hard':
            if clustering_method == "hdbscan":
                test_labels, _ = hdbscan.approximate_predict(clusterer, x_test)
                # change the labels to 0: Benign 1: Malicious
                test_labels[test_labels != -1] = 0   # in hdbscan every non -1 cluster is treated as normal
                test_labels[test_labels == -1] = 1
            elif clustering_method == "isolation forest":
                test_labels = clusterer.predict(x_test)
                # change the labels to 0: Benign 1: Malicious
                test_labels[test_labels == 1] = 0
                test_labels[test_labels == -1] = 1
            elif clustering_method == "LOF":
                test_labels = clusterer.predict(x_test)
                # change the labels to 0: Benign 1: Malicious
                test_labels[test_labels == 1] = 0
                test_labels[test_labels == -1] = 1
            else:
                # in the case of k-means the benign label is found by assuming that in the training set it is prevailing
                # so we set as the benign class the one with the most labels
                benign = 0 if clusterer.labels_[clusterer.labels_ == 0].size > \
                              clusterer.labels_[clusterer.labels_ == 1].size else 1
                # if the benign class is indeed the 0-th then we don't have to change anything since it is compatible
                # with our label mapping. Otherwise we have to flip the labels to be compatible
                test_labels = clusterer.predict(x_test)
                if benign != 0:
                    # change the labels to 0: Benign 1: Malicious
                    test_labels[test_labels == 0] = -1  # this line is necessary for the flipping to not mix labels
                    test_labels[test_labels == 1] = 0
                    test_labels[test_labels == -1] = 1
        else:
            if clustering_method == "hdbscan":
                # in this case an array of (number of samples, number of clusters) will be returned with the
                # probabilities of each sample belonging to each cluster
                test_labels = hdbscan.membership_vector(clusterer, x_test)
            elif clustering_method == "isolation forest":
                # in this case an array of (number of samples, 1) will be returned with the opposite of the anomaly
                # score for each sample
                test_labels = clusterer.score_samples(x_test)
            elif clustering_method == "LOF":
                # in this case an array of (number of samples, 1) will be returned with the opposite of the anomaly
                # score for each sample
                test_labels = clusterer.score_samples(x_test)
            else:
                # in this case an array of (number of samples, number of clusters) will be returned with the
                # distance of each sample from each cluster's center
                test_labels = clusterer.transform(x_test)
        return test_labels

    def fit_multivariate_gaussian(self):
        """
        Function for fitting a multivariate gaussian distribution on the the data points observed at each state/node
        :return: the estimated mean and covariance matrix of the fitted distribution
        """
        # features in rows and samples in columns
        x_train = np.transpose(self.attributes2dataset(self.observed_attributes).values)
        m = np.sum(x_train, axis=1) / x_train.shape[1]     # the estimated mean
        m = m.reshape([x_train.shape[0], 1])
        sigma = np.dot(x_train - m, (x_train - m).T) / x_train.shape[1]     # the estimated covariance matrix
        return m, sigma

    def predict_on_gaussian(self, m, sigma, epsilon=1e-16, prediction_type='hard'):
        """
        Function for predicting anomalies on the fitted multivariate gaussian distribution
        :param m: the estimated mean
        :param sigma: the estimated covariance matrix
        :param epsilon: the detection threshold
        :param prediction_type: the prediction type (hard or soft)
        :return: the prediction labels
        """
        x_test = np.transpose(self.attributes2dataset(self.testing_attributes).values)
        sigma_det = np.linalg.det(sigma)    # the determinant of the covariance matrix
        sigma_inv = np.linalg.inv(sigma)    # the inverse of the covariance matrix
        test_labels = np.array([np.asscalar(np.exp(-np.dot(np.dot((x_test[:, i].reshape([x_test.shape[0], 1]) - m).T,
                                                                  sigma_inv),
                                                           x_test[:, i].reshape([x_test.shape[0], 1]) - m) / 2)
                                            / ((2 * pi) ** (x_test.shape[0] / 2) * sqrt(sigma_det)))
                                for i in range(x_test.shape[1])])
        if prediction_type == 'hard':
            test_labels = (test_labels < epsilon).astype(np.int)
        return test_labels


class Model:
    def __init__(self):
        """
        Init function for the Model class used to initialize the dictionary containing the structure of the model
        """
        self.nodes_dict = {}

    def add_node(self, model_node):
        """
        Function that adds node in the structure dictionary of the model
        :param model_node: the node to be added
        :return: return code of success or failure
        """
        if model_node.label in self.nodes_dict.keys():
            return -1
        self.nodes_dict[model_node.label] = model_node
        return 0

    def remove_node(self, node_label):
        """
        Function that removes the specified node from the structure dictionary of the model
        :param node_label: the node to be removed
        :return: return code of success or failure
        """
        if node_label not in self.nodes_dict.keys():
            return -1
        del self.nodes_dict[node_label]
        return 0

    def fire_transition(self, src_node_label, input_attributes):
        """
        Function for finding the destination state given the source state and some input attributes' values
        :param src_node_label: the label of the source state
        :param input_attributes: the input attributes' values
        :return: the label of the destination state
        """
        # in case the source node is the root then only one choice is available
        if src_node_label == 'root':
            return self.nodes_dict[src_node_label].dst_nodes[0]
        # otherwise find the appropriate destination
        else:
            # in case there are no conditional transitions
            if len(self.nodes_dict[src_node_label].tran_conditions.keys()) == 0:
                # then if there are destination nodes
                if len(self.nodes_dict[src_node_label].dst_nodes) != 0:
                    # there should be only one otherwise there would be conditions around
                    if len(self.nodes_dict[src_node_label].dst_nodes) != 1:
                        print('Something went wrong -> Only one destination state should exist in non-conditional '
                              'cases!!!!')
                        return -1
                    # if there is indeed one return its label
                    else:
                        return self.nodes_dict[src_node_label].dst_nodes[0]
                # if there is no destination node then we are in a sink state so return the source label
                else:
                    return src_node_label
            # in case there are conditional transitions peak the appropriate one
            else:
                destinations = [(dst_node, self.nodes_dict[src_node_label].evaluate_transition(dst_node,
                                                                                               input_attributes))
                                for dst_node in self.nodes_dict[src_node_label].tran_conditions.keys()]
                return destinations[[destination[1] for destination in destinations].index(True)][0]

    def update_attributes(self, label, observed, attribute_type='train'):
        """
        Function for adding newly observed values for each attribute in the observed_attributes dict of each node
        :param label: the label of the node
        :param observed: the dictionary with the attributes' values to be added
        :param attribute_type: the type of the observed attributes ('train' | 'test')
        :return:
        """
        if attribute_type == 'train':
            for attribute, obs_value in observed.items():
                if len(self.nodes_dict[label].observed_attributes[attribute]) == 0:
                    self.nodes_dict[label].observed_attributes[attribute] = [obs_value]
                else:
                    self.nodes_dict[label].observed_attributes[attribute] += [obs_value]
        else:
            for attribute, obs_value in observed.items():
                if len(self.nodes_dict[label].testing_attributes[attribute]) == 0:
                    self.nodes_dict[label].testing_attributes[attribute] = [obs_value]
                else:
                    self.nodes_dict[label].testing_attributes[attribute] += [obs_value]

    def update_indices(self, label, ind, attribute_type='train'):
        """
        Function for adding a newly observed index for each observed record added in the given node
        :param label: the label of the node
        :param ind: an integer indicating the index to be added
        :param attribute_type: the type of the observed attributes ('train' | 'test')
        :return:
        """
        if attribute_type == 'train':
            self.nodes_dict[label].observed_indices += [ind]
        else:
            self.nodes_dict[label].testing_indices += [ind]

    def reset_attributes(self, attribute_type='train'):
        """
        Function for resetting all attributes (observed or testing) values in all nodes
        :param attribute_type: the type of the observed attributes ('train' | 'test')
        :return:
        """
        if attribute_type == 'train':
            for node_label in self.nodes_dict.keys():
                self.nodes_dict[node_label].reset_observed_attributes()
        else:
            for node_label in self.nodes_dict.keys():
                self.nodes_dict[node_label].reset_testing_attributes()

    def reset_indices(self, attribute_type='train'):
        """
        Function for resetting all indices (observed or testing) values in all nodes
        :param attribute_type: the type of the observed attributes ('train' | 'test')
        :return:
        """
        if attribute_type == 'train':
            for node_label in self.nodes_dict.keys():
                self.nodes_dict[node_label].reset_observed_indices()
        else:
            for node_label in self.nodes_dict.keys():
                self.nodes_dict[node_label].reset_testing_indices()

    def get_maximum_weight(self):
        """
        Function for finding the maximum prediction weight of the model as the maximum number of observations in any of
        its states. This weight is used when weighted prediction is applied.
        :return: the maximum prediction weight
        """
        maximum_weight = 0
        for node_label in self.nodes_dict.keys():
            if len(self.nodes_dict[node_label].observed_indices) > maximum_weight:
                maximum_weight = len(self.nodes_dict[node_label].observed_indices)
        return maximum_weight

    def set_all_weights(self, maximum_weight):
        """
        Function for setting the weights in each state of the model. The weight of each state is set as the number of
        observations of the state divided by the maximum weight
        :param maximum_weight: the maximum weight of the model
        :return:
        """
        for node_label in self.nodes_dict.keys():
            self.nodes_dict[node_label].weight = len(self.nodes_dict[node_label].observed_indices) / maximum_weight
