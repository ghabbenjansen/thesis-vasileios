import operator
import numpy as np
import pandas as pd
import hdbscan
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from math import pi, sqrt


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

    def evaluate_transition(self, dst_node_label, input_attributes):
        """
        Function for evaluating if a transition from a source to a destination state under the provided input attributes'
        values can be fired
        :param dst_node_label: the label of the destination state
        :param input_attributes: the input attributes' values
        :return: a boolean value denoting the ability to fire the transition
        """
        # TODO: check validity in case there are no transition conditions
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

    def fit_clusters_on_observed(self, clustering_method='k-means'):
        """
        Function for fitting clusters on the data points observed at each state/node
        :param clustering_method: the clustering method (currently k-means | hdbscan | Isolation Forest | LOF)
        :return: the fitted cluster estimator
        """
        x_train = self.attributes2dataset(self.observed_attributes)
        if clustering_method == "hdbscan":
            # TODO: maybe add an outlier detection layer (GLOSH) before the actual clustering
            clusterer = hdbscan.HDBSCAN(min_cluster_size=20, metric='manhattan', prediction_data=True). \
                fit(RobustScaler().fit_transform(x_train.values))
        elif clustering_method == "isolation-forest":
            clusterer = IsolationForest().fit(x_train.values)
        elif clustering_method == "LOF":
            clusterer = LocalOutlierFactor(metric='manhattan', novelty=True).fit(x_train.values)
        else:
            clusterer = KMeans(n_clusters=2).fit(x_train.values)
        return clusterer

    def predict_on_clusters(self, clusterer, clustering_method='k-means', clustering_type='hard'):
        """
        Function for predicting the cluster labels on the testing traces of the node given a fitted cluster estimator
        :param clusterer: the fitted cluster estimator
        :param clustering_method: the clustering method (currently k-means | hdbscan | Isolation Forest | LOF)
        :param clustering_type: the clustering type to be used (hard or soft)
        :return: the predicted labels
        """
        x_test = self.attributes2dataset(self.testing_attributes)
        if clustering_type == 'hard':
            if clustering_method == "hdbscan":
                test_labels, _ = hdbscan.approximate_predict(clusterer, x_test.values)
            elif clustering_method == "isolation-forest":
                test_labels = clusterer.predict(x_test.values)
            elif clustering_method == "LOF":
                test_labels = clusterer.predict(x_test.values)
            else:
                test_labels = clusterer.predict(x_test.values)
        else:
            if clustering_method == "hdbscan":
                # in this case an array of (number of samples, number of clusters) will be returned with the
                # probabilities of each sample belonging to each cluster
                test_labels = hdbscan.membership_vector(clusterer, x_test.values)
            elif clustering_method == "isolation-forest":
                # in this case an array of (number of samples, 1) will be returned with the opposite of the anomaly
                # score for each sample
                test_labels = clusterer.score_samples(x_test.values)
            elif clustering_method == "LOF":
                # in this case an array of (number of samples, 1) will be returned with the opposite of the anomaly
                # score for each sample
                test_labels = clusterer.score_samples(x_test.values)
            else:
                # in this case an array of (number of samples, number of clusters) will be returned with the distance of
                # each sample from each cluster's center
                test_labels = clusterer.transform(x_test.values)
        return test_labels

    def fit_multivariate_gaussian(self):
        """
        Function for fitting a multivariate gaussian distribution on the the data points observed at each state/node
        :return: the estimated mean and covariance matrix of the fitted distribution
        """
        # features in rows and samples in columns
        x_train = np.transpose(self.attributes2dataset(self.observed_attributes).values)
        m = np.mean(x_train, axis=1) / x_train.shape[1]     # the estimated mean
        m = m.reshape([x_train.shape[0], 1])
        sigma = np.dot(x_train - m, (x_train - m).T) / x_train.shape[1]     # the estimated covariance matrix
        return m, sigma

    def predict_on_gaussian(self, m, sigma, epsilon=0.01, prediction_type='hard'):
        """
        Function for predicting anomalies on the fitted multivariate gaussian distribution
        :param m: the estimated mean
        :param sigma: the estimated covariance matrix
        :param epsilon: the detection threshold
        :param prediction_type: the prediction type (hard or soft)
        :return: the prediction labels
        """
        x_test = np.transpose(self.attributes2dataset(self.testing_attributes))
        sigma_det = np.linalg.det(sigma)    # the determinant of the covariance matrix
        sigma_inv = np.linalg.inv(sigma)    # the inverse of the covariance matrix
        test_labels = np.exp(-np.dot(np.dot((x_test - m).T, sigma_inv), x_test - m) / 2) / \
                      ((2 * pi) ** (x_test.shape[0] / 2) * sqrt(sigma_det))
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
        destinations = [(dst_node, self.nodes_dict[src_node_label].evaluate_transition(dst_node, input_attributes))
                        for dst_node in self.nodes_dict[src_node_label].tran_conditions.keys()]
        return destinations[np.where(destinations)[0][0]][0]

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
                self.nodes_dict[label].observed_attributes[attribute] += obs_value
        else:
            for attribute, obs_value in observed.items():
                self.nodes_dict[label].testing_attributes[attribute] += obs_value

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
