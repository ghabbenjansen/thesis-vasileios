import operator
import numpy as np


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

    def attributes2dataset(self):
        """
        Function for converting the observed attributes dict to a numpy array to be processed
        like a dataset (m examples x n attributes)
        :return: the dataset as a numpy array
        """
        return np.array(list(self.observed_attributes.values())).transpose()


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

    def update_attributes(self, label, observed):
        """
        Function for adding newly observed values for each attribute in the observed_attributes dict of each node
        :param label: the label of the node
        :param observed: the dictionary with the attributes' values to be added
        :return:
        """
        for attribute, obs_value in observed.items():
            self.nodes_dict[label].observed_attributes[attribute] += obs_value

    def reset_attributes(self):
        """
        Function for resetting all observed_attributes values in all nodes
        :return:
        """
        for node_label in self.nodes_dict.keys():
            self.nodes_dict[node_label].reset_observed_attributes()
