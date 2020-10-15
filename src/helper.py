#!/usr/bin/python

from pandas.tseries.offsets import DateOffset
from copy import deepcopy
from scipy.stats import mode
from statistics import mean
from src.model import ModelNode, Model
from tslearn.metrics import dtw
from sklearn.preprocessing import MinMaxScaler
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
from collections import defaultdict


def reduce_data_by_label(data, threshold, flag):
    """
    Function for reducing the number of data points in the input dataframe in each class.
    :param data: the input dataframe
    :param threshold: the threshold above which data points are cut
    :param flag: the dataset-dependent flag
    :return: the reduced dataframe
    """
    if flag == 'CTU-bi':
        normal = data[data['label'].str.contains("From-Normal")].sort_values(by='date').reset_index(drop=True)
        anomalous = data[data['label'].str.contains("From-Botnet")].sort_values(by='date').reset_index(drop=True)
    elif flag == 'UNSW':
        normal = data[data['label'] == 0].sort_values(by='date').reset_index(drop=True)
        anomalous = data[data['label'] == 1].sort_values(by='date').reset_index(drop=True)
    else:
        normal = data[data['label'] == 'BENIGN'].sort_values(by='date').reset_index(drop=True)
        anomalous = data[data['label'] != 'BENIGN'].sort_values(by='date').reset_index(drop=True)

    if normal.shape[0] > threshold:
        normal = normal.iloc[:threshold]
    if anomalous.shape[0] > threshold:
        anomalous = anomalous.iloc[:threshold]

    return pd.concat([normal, anomalous], ignore_index=True).sort_values(by='date').reset_index(drop=True)


def keep_only_major_connection(data):
    """
    Function for keeping only the major connection for a host
    :param data: the host data
    :return: the host data containing only the flows with the major destination IP
    """
    major_dst_ip = data.groupby('dst_ip').size().idxmax(axis=0)
    return data[data['dst_ip'] == major_dst_ip].sort_values(by='date').reset_index(drop=True)


def check_ips(x, ips_dict):
    """
    Helper function for swaping the needed values when bidirectional flows are examined. This function is meant
    to be applied on each row a dataframe.
    :param x: input row of the dataframe
    :param ips_dict: a dictionary with the ips seen, when only the source IP is considered as host, and their counts
    :return: the transformed row
    """
    # if the destination IP has been seen as a host and has more flows than the current source IP, make the swap
    if x['dst_ip'] in ips_dict.keys() and ips_dict[x['dst_ip']] > ips_dict[x['src_ip']]:
        x['src_ip'], x['dst_ip'] = x['dst_ip'], x['src_ip']
        x['src_port'], x['dst_port'] = x['dst_port'], x['src_port']
        x['src_bytes'], x['dst_bytes'] = x['dst_bytes'], x['src_bytes']
    return x


def select_hosts(init_data, threshold=50, bidirectional=False):
    """
    Function for identifying only the flows of source IPs with at least a threshold number of records in the data.
    :param init_data: the initial data
    :param threshold: the threshold number of flows per host IP
    :param bidirectional: a boolean flag for checking for host IPs in both directions (source and destination). If set
    to False, only source IPs will be considered as hosts
    :return: the selected data
    """
    host_cnts = init_data.groupby(by='src_ip').size().reset_index().rename(columns={0: "size"})
    if bidirectional:
        ips_counts = dict(host_cnts.values.tolist())
        init_data = init_data.apply(lambda x: check_ips(x, ips_counts), axis=1)
        host_cnts = init_data.groupby(by='src_ip').size().reset_index().rename(columns={0: "size"})
    return host_cnts[host_cnts['size'] > threshold]


def check_connections(x, connections_dict):
    """
    Helper function for swaping the needed values when bidirectional connections are examined. This function is meant
    to be applied on each row a dataframe.
    :param x: input row of the dataframe
    :param connections_dict: a dictionary with the connections seen in the forward direction and their counts
    :return: the transformed row
    """
    connection = x['src_ip'] + '-' + x['dst_ip']
    connection_reversed = x['dst_ip'] + '-' + x['src_ip']
    # if the connection in the forward direction has been seen as a forward connection and has more flows than the
    # current forward connection, make the swap
    if connection_reversed in connections_dict.keys() and connections_dict[connection_reversed] > \
            connections_dict[connection]:
        x['src_ip'], x['dst_ip'] = x['dst_ip'], x['src_ip']
        x['src_port'], x['dst_port'] = x['dst_port'], x['src_port']
        x['src_bytes'], x['dst_bytes'] = x['dst_bytes'], x['src_bytes']
    return x


def select_connections(init_data, threshold=50, bidirectional=False):
    """
    Function for identifying only the flows with at least a threshold number of source-destination IP pairs in the data.
    :param init_data: the initial data
    :param threshold: the threshold number of flows per source-destination IP pairs
    :param bidirectional: a boolean flag for checking for connections in both directions (source and destination). If
    set to False, only the original direction will be checked
    :return: the selected data
    """
    connections_cnts = init_data.groupby(['src_ip', 'dst_ip']).size().reset_index().rename(columns={0: "size"})
    if bidirectional:
        connections_counts = dict([[triple[0] + '-' + triple[1], triple[2]] for triple in connections_cnts[['src_ip', 'dst_ip', 'size']].values.tolist()])
        init_data = init_data.apply(lambda x: check_connections(x, connections_counts), axis=1)
        connections_cnts = init_data.groupby(['src_ip', 'dst_ip']).size().reset_index().rename(columns={0: "size"})
    return connections_cnts[connections_cnts['size'] > threshold]


def set_windowing_vars(data):
    """
    Function for automatically calculating an initial estimation of the time windows and strides to be used for creating
    the traces, as the median of the time differences between the flows in the dataframe
    :param data: the input dataframe
    :return: a tuple with the calculated time windows and strides in a dataframe format
    """
    # find the median of the time differences in the dataframe
    if data.shape[0] != 1:
        median_diff = data['date'].sort_values().diff().median()
        return 25 * median_diff, 5 * median_diff
    # in case there is only one flow in the dataset just return zero-length Timedelta results
    else:
        return pd.to_timedelta('0s'), pd.to_timedelta('0s')


def find_percentile(val, percentiles):
    """
    Helper function returning the relative index of placement in the percentiles
    :param val: the value to be indexed
    :param percentiles: the percentile limits
    :return: the index of val in the percentiles
    """
    ind = len(percentiles)
    for i, p in enumerate(percentiles):
        if val <= p:
            ind = i
            break
    return ind


def check_existence(val, seen):
    """
    Helper function for checking if a value is in a list of seen values. If this is the case, the index of the value in
    the list is returned. Otherwise, the length of the list is returned
    :param val: the value to be searched
    :param seen: the list of seen values
    :return: the needed output given the result of the condition checked
    """
    return seen.index(val) if val in seen else len(seen)


def find_discretization_clusters(data, selected):
    """
    Function for applying the ELBOW method to a number of selected features so that the appropriate number of clusters
    for each feature can be identified and used as discretization limits for each feature.
    :param data: the input dataframe
    :param selected: the selected features
    :return: a dictionary with the selected number of discretization limits for each feature
    """
    discretization_limits = {}
    for sel in selected:
        # apply the elbow method
        print('----------------------- Finding optimal number of bins for {} -----------------------'.format(sel))
        Sum_of_squared_distances = []
        for k in range(1, 11):
            km = KMeans(n_clusters=k)
            km = km.fit(data[sel].values.reshape(-1, 1))
            Sum_of_squared_distances.append(km.inertia_)

        plt.figure()
        plt.plot(range(1, 11), Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.grid()
        plt.show()

        # provide the desired number of discretization points according to the ELBOW plot
        percentile_num = int(input('Enter your preferred number of discretization points: '))
        # and find the percentile limits for the examined feature
        discretization_limits[sel] = list(map(lambda p: np.percentile(data[sel], p), 100 *
                                              np.arange(0, 1, 1 / percentile_num)[1:]))
    return discretization_limits


def netflow_encoding(data, selected, discretization_dict):
    """
    Function for calculating the encoding of each flow according to the method proposed in the paper
    "Learning Behavioral Fingerprints From Netflows Using Timed Automata"
    :param data: the dataframe with the input data
    :param selected: a list with the selected features
    :param discretization_dict: the dictionary with the discritization information
    :return:
    """
    space_size = 1
    # first calculate the total space size of the encoding
    for k, v in discretization_dict.items():
        space_size *= (len(v)+1)
    # then created the encoded aggregation of all features
    data['encoding'] = 0
    for sel in selected:
        sel_name = sel[:-4] if sel not in discretization_dict.keys() else sel
        size = len(discretization_dict[sel_name]) + 1
        data['encoding'] += data[sel] * space_size / size
        space_size /= size
    data['encoding'] = data['encoding'].astype(int)


def traces_dissimilarity(trace1, trace2, multivariate=True, normalization=True):
    """
    Function for calculating the dissimilarity between two input traces. The traces are in the form of list of lists and
    are dealt either as multivariate series or as multiple univariate series depending on the value of the multivariate
    flag provided
    :param trace1: the first trace
    :param trace2: the second trace
    :param multivariate: the multivariate flag
    :param normalization: the normalization flag for performing (or not) min-max normalization
    :return: the dissimilarity score (the lower the score the higher the similarity)
    """
    if normalization:
        traces = MinMaxScaler().fit_transform(trace1 + trace2)
        trace1 = traces[:len(trace1)].tolist()
        trace2 = traces[len(trace1):].tolist()
    return dtw(trace1, trace2) if multivariate else mean([dtw(list(list(zip(*trace1))[j]), list(list(zip(*trace2))[j]))
                                                          for j in range(len(trace1[0]))])


def convert2flexfringe_format(win_data, ints=True):
    """
    Function to convert the windowed data into a trace in the format accepted by the multivariate version of flexfringe
    :param win_data: the windowed dataframe
    :param ints: flag showing if there are only int data in the dataframe
    :return: a list of the events in the trace with features separated by comma in each event
    """
    fun = lambda x: int(x) if ints else float(x)
    return list(map(lambda x: ','.join(map(lambda t: str(fun(t)), x)), win_data.to_numpy().tolist()))


def trace2list(trace):
    """
    Function for converting a list of string records of a trace to a list of lists
    :param trace: the list with the string records
    :return: the converted list of lists
    """
    return list(map(lambda x: list(map(int, x.split(','))), trace))


def calculate_window_mask(data, start_date, end_date):
    """
    Function for calculating the window mask for the input dataframe given a starting and an ending date
    :param data: the input dataframe
    :param start_date: the starting date
    :param end_date: the ending date
    :return: the window mask
    """
    return (data['date'] >= start_date) & (data['date'] <= end_date)


def aggregate_in_windows(data, selected_features, window, timed=False, resample=False, new_features=True):
    """
    Function for aggregating specific features of a dataframe in rolling windows of length window
    Currently the following features are taken into account: source port, destination ip/port, originator's bytes,
    responder's bytes, duration, and protocol
    :param data: the input dataframe
    :param selected_features: the features that are contained in the dataframe (this value is passed even if it can be
    inferred by the columns of the dataframe for ordering purposes between different runs of the function)
    :param window: the window length
    :param timed: boolean flag specifying if aggregation window should take into account the timestamps
    :param resample: boolean flag specifying if aggregation window should be rolling or resampling
    :param new_features: boolean flag specifying if new features should be added to the existing ones
    :return: a dataframe with the aggregated features
    """
    old_column_names = deepcopy(selected_features)
    # if the timed flag is True then timestamps are used as indices
    if timed:
        data.set_index('date', inplace=True)
    if not resample:
        for feature in old_column_names:
            # check for ports in features
            if 'port' in feature:
                if new_features:
                    data['unique_' + feature + 's'] = data[feature].rolling(window, min_periods=1).apply(lambda x:
                                                                                                         len(set(x)),
                                                                                                         raw=False)
                    data['std_' + feature + 's'] = data[feature].rolling(window, min_periods=1).std()
                else:
                    data['median_' + feature] = data[feature].rolling(window, min_periods=1).median()
            # check for protocol
            if 'protocol_num' in feature:
                data['argmax_protocol_num'] = data['protocol_num'].rolling(window, min_periods=1). \
                    apply(lambda x: mode(x)[0], raw=False)
                if new_features:
                    data['std_protocol_num'] = data['protocol_num'].rolling(window, min_periods=1).std()
            # check for encoding in case of discretized input
            if 'encoding' in feature:
                data['argmax_encoding'] = data['encoding'].rolling(window, min_periods=1). \
                    apply(lambda x: mode(x)[0], raw=False)
                if new_features:
                    data['std_encoding'] = data['encoding'].rolling(window, min_periods=1).std()
            # check for duration in features
            if 'duration' in feature:
                data['median_' + feature] = data[feature].rolling(window, min_periods=1).median()
                if new_features:
                    data['std_' + feature] = data[feature].rolling(window, min_periods=1).std()
            # check for bytes in features
            if 'bytes' in feature:
                data['median_' + feature] = data[feature].rolling(window, min_periods=1).median()
                if new_features:
                    data['std_' + feature] = data[feature].rolling(window, min_periods=1).std()
            # check for destination IP in features in case new features are considered
            if 'dst_ip' in feature:
                if new_features:
                    data['unique_dst_ips'] = pd.DataFrame(pd.Categorical(data['dst_ip']).codes, index=data.index). \
                        rolling(window, min_periods=1).apply(lambda x: len(set(x)), raw=False)
        data.drop(columns=old_column_names, inplace=True)
        data.bfill(axis='rows', inplace=True)
    else:
        # can be called only if timed flag has been set to True
        frames = []
        new_column_names = []
        # check for ports in features
        for feature in old_column_names:
            if 'port' in feature:
                new_column_names += (['unique' + feature + 's', 'std_' + feature + 's'] if new_features else
                                     ['median_' + feature])
                frames += ([data[feature].resample(window).nunique(), data[feature].resample(window).std()] if
                           new_features else [data[feature].resample(window).median()])
            # check for protocol
            if 'protocol_num' in feature:
                new_column_names += (['argmax_protocol_num', 'std_protocol_num'] if new_features else
                                     ['argmax_protocol_num'])
                frames += ([data['protocol_num'].resample(window).apply(lambda x: mode(x)[0]),
                            data['protocol_num'].resample(window).std()] if new_features else
                           [data['protocol_num'].resample(window).apply(lambda x: mode(x)[0])])
            # check for encoding in case of discretized input
            if 'encoding' in feature:
                new_column_names += (['argmax_encoding', 'std_encoding'] if new_features else
                                     ['argmax_encoding'])
                frames += ([data['encoding'].resample(window).apply(lambda x: mode(x)[0]),
                            data['encoding'].resample(window).std()] if new_features else
                           [data['encoding'].resample(window).apply(lambda x: mode(x)[0])])
            # check for duration in features
            if 'duration' in feature:
                new_column_names += (['median_' + feature, 'std_' + feature] if new_features else ['median_' + feature])
                frames += ([data[feature].resample(window).median(), data[feature].resample(window).std()] if
                           new_features else [data[feature].resample(window).median()])
            # check for bytes in features
            if 'bytes' in feature:
                new_column_names += (['median_' + feature, 'std_' + feature] if new_features else ['median_' + feature])
                frames += ([data[feature].resample(window).median(), data[feature].resample(window).std()]
                           if new_features else [data[feature].resample(window).median()])
            # check for destination IP in features in case new features are considered
            if 'dst_ip' in feature:
                if new_features:
                    new_column_names += ['unique_dst_ips']
                    frames += [data['dst_ip'].resample(window).nunique()]
        data = pd.concat(frames, axis=1)
        data.columns = new_column_names
        data.dropna(inplace=True)
        # handle the case of discretized data in which the dropna is not sufficient by itself
        if len(new_column_names) == 1:
            data = data[data.astype(str)[new_column_names[0]] != '[]']
    return data


def aggregate_static(data, selected_features, window, new_features=True):
    """
    Function for aggregating specific features of a dataframe in static non-timed windows of length window
    Currently the following features are taken into account: source port, destination ip/port, originator's bytes,
    responder's bytes, duration, and protocol
    :param data: the input dataframe
    :param selected_features: the features that are contained in the dataframe (this value is passed even if it can be
    inferred by the columns of the dataframe for ordering purposes between different runs of the function)
    :param window: the window length
    :param new_features: boolean flag specifying if new features should be added to the existing ones
    :return: a dataframe with the aggregated features
    """
    old_column_names = deepcopy(selected_features)
    # can be called only if timed flag has been set to True
    frames = []
    new_column_names = []
    # check for ports in features
    for feature in old_column_names:
        if 'port' in feature:
            new_column_names += (['unique' + feature + 's', 'std_' + feature + 's'] if new_features else
                                 ['median_' + feature])
            frames += ([data[feature].groupby(data.index // window).nunique(),
                        data[feature].groupby(data.index // window).std()] if
                       new_features else [data[feature].groupby(data.index // window).median()])
        # check for protocol
        if 'protocol_num' in feature:
            new_column_names += (['argmax_protocol_num', 'std_protocol_num'] if new_features else
                                 ['argmax_protocol_num'])
            frames += ([data['protocol_num'].groupby(data.index // window).apply(lambda x: mode(x)[0][0]),
                        data['protocol_num'].groupby(data.index // window).std()] if new_features else
                       [data['protocol_num'].groupby(data.index // window).apply(lambda x: mode(x)[0][0])])
        # check for encoding in case of discretized input
        if 'encoding' in feature:
            new_column_names += (['argmax_encoding', 'std_encoding'] if new_features else
                                 ['argmax_encoding'])
            frames += ([data['encoding'].groupby(data.index // window).apply(lambda x: mode(x)[0][0]),
                        data['encoding'].groupby(data.index // window).std()] if new_features else
                       [data['encoding'].groupby(data.index // window).apply(lambda x: mode(x)[0][0])])
        # check for duration in features
        if 'duration' in feature:
            new_column_names += (['median_' + feature, 'std_' + feature] if new_features else ['median_' + feature])
            frames += ([data[feature].groupby(data.index // window).median(),
                        data[feature].groupby(data.index // window).std()] if
                       new_features else [data[feature].groupby(data.index // window).median()])
        # check for bytes in features
        if 'bytes' in feature:
            new_column_names += (['median_' + feature, 'std_' + feature] if new_features else ['median_' + feature])
            frames += ([data[feature].groupby(data.index // window).median(),
                        data[feature].groupby(data.index // window).std()]
                       if new_features else [data[feature].groupby(data.index // window).median()])
        # check for destination IP in features in case new features are considered
        if 'dst_ip' in feature:
            if new_features:
                new_column_names += ['unique_dst_ips']
                frames += [data['dst_ip'].groupby(data.index // window).nunique()]
    data = pd.concat(frames, axis=1)
    data.columns = new_column_names
    data.dropna(inplace=True)
    # handle the case of discretized data in which the dropna is not sufficient by itself
    if len(new_column_names) == 1:
        data = data[data.astype(str)[new_column_names[0]] != '[]']
    return data


def extract_traces_from_window(data, selected, window, stride, trace_limits, total, progress_list,
                               dynamic=True, aggregation=False, resample=False, new_features=True):
    """
    Function for extracting traces from the imput dataframe. The features to be taken into account are provided in the
    selected list. Each trace is extracted by rolling a window of window seconds in the input data with a stride of
    stride seconds. If dynamic flag is set to True, then a dynamically changing window is used instead. If aggregation
    flag is set to True, then aggregation windows are created in each rolling window. If resample flag is set to True,
    then instead of a rolling window a resampling one is used during aggregation. If new_features is set to True, then
    in each aggregation windows are used, instead of creating new features only an aggregated view of the existing ones
    is maintained.
    :param data: the input dataframe
    :param selected: the features to be used
    :param window: the window size
    :param stride: the stride size
    :param trace_limits: a tuple containing the minimum and maximum length that a trace can have
    :param total: total number of flows in the original dataframe (for progress visualization purposes)
    :param progress_list: list with the progress in processing the original dataframe (for progress visualization
    purposes)
    :param dynamic: boolean flag about the use of dynamically changing windows
    :param aggregation: the aggregation flag - if set to True, then aggregation windows are created
    :param resample: the resampling flag - if set to True, then resampling is used in the aggregation windows
    :param new_features: boolean flag specifying if new features should be added to the existing ones
    :return: the traces extracted in a list, the indices of each trace in a list, and the number of features extracted
    """

    # create an anonymous function for increasing timestamps given the type of the window (int or Timedelta)
    time_inc = lambda x, w: x + DateOffset(seconds=w) if type(window) == int else x + w
    # obtain the indices residing in the processed data
    data_indices = data.index.tolist()
    # set the initial start and end dates, as well as the empty traces' list and the window limits
    start_date = data['date'].iloc[0]
    end_date = time_inc(start_date, window)
    traces = []  # list of lists
    traces_indices = []  # list of lists for storing the indices of the flows contained in each trace
    # the minimum and maximum indices of the time window under consideration
    # two values are used for the indices of two consecutive windows
    min_idx = [-2, -1]
    max_idx = [-2, -1]
    # structures just for progress visualization purposes
    cnt = 0
    # extract the traces' limits
    min_trace_length, max_trace_length = trace_limits
    # create a dict for testing if all the flows have been included in the traces
    assertion_dict = dict(zip(data_indices, len(data_indices) * [False]))
    # keep a copy of the actually selected features in case aggregation is used
    old_selected = deepcopy(selected)
    # keep also a variable of the number of features to be used for the model creation
    num_of_features = len(selected)
    # one-time flag for the case that the first time window is proven to be too large
    first_large = True
    # iterate through the input dataframe until the end date is greater than the last date recorded
    while end_date < data['date'].iloc[-1]:
        # retrieve the window of interest
        time_mask = calculate_window_mask(data, start_date, end_date)
        windowed_data = data[time_mask]
        window_len = len(windowed_data.index.tolist())
        # if there is at least one record in the window
        if window_len != 0:
            if dynamic:
                # store the minimum and maximum indices of the time window to evaluate how much it moved
                if min_idx[0] == -2:  # the case of the first recorded time window
                    min_idx[0] = data.index[time_mask].tolist()[0]
                    max_idx[0] = data.index[time_mask].tolist()[-1]
                elif min_idx[1] == -1:  # the case of the second recorded time window
                    min_idx[1] = data.index[time_mask].tolist()[0]
                    max_idx[1] = data.index[time_mask].tolist()[-1]
                else:  # otherwise update the previous values and add the new ones
                    min_idx[0] = deepcopy(min_idx[1])
                    max_idx[0] = deepcopy(max_idx[1])
                    min_idx[1] = data.index[time_mask].tolist()[0]
                    max_idx[1] = data.index[time_mask].tolist()[-1]

                # first check if the time window captured new information
                while min_idx[0] == min_idx[1] and max_idx[0] == max_idx[1]:
                    print('-------------- No change between traces ==> Increasing the stride... --------------')
                    start_date = time_inc(start_date, stride)
                    end_date = time_inc(start_date, window)
                    time_mask = calculate_window_mask(data, start_date, end_date)
                    window_len = len(data[time_mask].index.tolist())
                    # if the new window is empty or we have surpassed the next unseen flow, the next window is set to
                    # start at the timestamp of this unseen flow
                    if window_len == 0 or data.index[time_mask].tolist()[0] > max_idx[1] + 1:
                        start_date = data['date'].loc[max_idx[1] + 1]
                        end_date = time_inc(start_date, window)
                        time_mask = calculate_window_mask(data, start_date, end_date)
                        window_len = len(data[time_mask].index.tolist())
                    # set the updated indices
                    min_idx[1] = data.index[time_mask].tolist()[0]
                    max_idx[1] = data.index[time_mask].tolist()[-1]
                    # and increase the stride in case we still haven't captured new information
                    if min_idx[0] == min_idx[1] and max_idx[0] == max_idx[1]:
                        stride *= 2
                        if stride >= window:
                            window = stride * 5

                # set the parameters for the length adjustment process of each trace
                init_magnifier = 2
                magnifier = 2
                reducer = 0.05
                # check that the trace length conforms to the specified limits
                while window_len < min_trace_length or window_len > max_trace_length:
                    # first check the case of a very large window
                    while window_len > max_trace_length:
                        print('-------------- Too many flows in the trace ==> Reducing time window... --------------')
                        window /= magnifier
                        if stride >= window:
                            stride = window / 5
                        end_date = time_inc(start_date, window)
                        time_mask = calculate_window_mask(data, start_date, end_date)
                        window_len = len(data[time_mask].index.tolist())

                    # then check the case of a very small window
                    while window_len < min_trace_length:
                        print('-------------- Too few flows in the trace ==> Increasing time window... --------------')
                        window *= magnifier
                        end_date = time_inc(start_date, window)
                        time_mask = calculate_window_mask(data, start_date, end_date)
                        window_len = len(data[time_mask].index.tolist())
                        # limit case to prevent integer overflow in the window size
                        if end_date > data['date'].iloc[-1]:
                            break

                    # and update the window indices
                    if first_large:
                        min_idx[0] = data.index[time_mask].tolist()[0]
                        max_idx[0] = data.index[time_mask].tolist()[-1]
                    else:
                        min_idx[1] = data.index[time_mask].tolist()[0]
                        max_idx[1] = data.index[time_mask].tolist()[-1]

                    # update the magnifier in case more iterations are needed due to fluctuations
                    magnifier -= reducer
                    magnifier = round(magnifier, len(str(reducer).split('.')[1]))
                    # in case that the fluctuations cannot be dealt with the current values, refine them and start over
                    if magnifier <= 1:
                        magnifier = init_magnifier + 1
                        reducer = reducer / 2
                    # limit case to prevent endless loop
                    if end_date > data['date'].iloc[-1]:
                        break

                # set the one-time flag of the first window to False
                first_large = False
                # finally get the current window
                windowed_data = data[time_mask]

            # insert the indices of the current trace to the assertion dictionary
            assertion_dict.update(zip(data.index[time_mask].tolist(), len(data.index[time_mask].tolist()) * [True]))

            # create aggregated features if needed (currently with a hard-coded window length)
            if aggregation:
                # this checks also if the window is big enough. If not then it sets the aggregation window to 1/5 of
                # the window
                # aggregation_length = min(Timedelta(seconds=5), (min(end_date, data['date'].iloc[-1])-start_date)/5) \
                #     if resample else min(10, int(len(windowed_data.index)))
                aggregation_length = '5S' if resample else min(10, int(len(windowed_data.index))) # old version
                timed = True if resample else False
                windowed_data = aggregate_in_windows(windowed_data[selected].copy(deep=True), selected,
                                                     aggregation_length, timed, resample, new_features)
                selected = windowed_data.columns.values
                num_of_features = len(selected)

            # extract the trace of this window and add it to the traces' list
            ints = False if (aggregation and 'encoding' not in old_selected) or 'duration' in old_selected else True
            # this case applies only on resampling in case there are no more than 1 flow per resampling window
            if windowed_data.shape[0] != 0:
                traces += [convert2flexfringe_format(windowed_data[selected], ints)]
            selected = deepcopy(old_selected)
            # store also the flow indices of the current time window
            if windowed_data.shape[0] != 0:  # this case applies only on resampling as explained above
                traces_indices += [windowed_data.index.tolist()]

            # update the progress variable
            cnt = data.index[time_mask].tolist()[-1]

            # increment the window limits
            start_date = time_inc(start_date, stride)
            end_date = time_inc(start_date, window)
        # if there are no records in the window
        else:
            if dynamic:
                # if there is no observation in the window just set as start date the one of the next non-visited index
                print('------------- No records in the trace ==> Proceeding to the next recorded flow... -------------')
                # check if there are less than two windows captured
                if max_idx[1] < 0:
                    # mostly to catch an implementation error since at least the first window should have records
                    if max_idx[0] < 0:
                        print('This should not happen!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        start_date = data['date'].iloc[0]
                    else:
                        start_date = data['date'].loc[max_idx[0] + 1]
                # otherwise set the start date of the last visited index + 1
                else:
                    start_date = data['date'].loc[max_idx[1] + 1]
                end_date = time_inc(start_date, window)
            else:
                # increment the window limits
                start_date = time_inc(start_date, stride)
                end_date = time_inc(start_date, window)

        # show progress
        prog = int((cnt / total) * 100)
        if prog // 10 != 0 and prog // 10 not in progress_list:
            progress_list += [prog // 10]
            print('More than ' + str((prog // 10) * 10) + '% of the data processed...')

    # addition of the last flows in the dataframe in case they weren't added
    if not all(list(assertion_dict.values())):
        time_mask = calculate_window_mask(data, start_date, end_date)
        windowed_data = data[time_mask]
        # in case that the start date is also greater than the last seen flow then set the start date appropriately
        if windowed_data.index.tolist()[0] > max_idx[1] + 1:
            if max_idx[1] < 0:
                if max_idx[0] < 0:
                    start_date = data['date'].iloc[0]
                else:
                    start_date = data['date'].loc[max_idx[0] + 1]
            else:
                start_date = data['date'].loc[max_idx[1] + 1]
            time_mask = calculate_window_mask(data, start_date, end_date)
            windowed_data = data[time_mask]
        # update the assertion dictionary
        assertion_dict.update(zip(data.index[time_mask].tolist(), len(data.index[time_mask].tolist()) * [True]))
        # check for aggregation
        if aggregation:
            # this checks also if the window is big enough. If not then it sets the aggregation window to 1/5 of
            # the window
            # aggregation_length = min(Timedelta(seconds=5), (min(end_date, data['date'].iloc[-1]) - start_date) / 5) \
            #     if resample else min(10, int(len(windowed_data.index)))
            aggregation_length = '5S' if resample else min(10, int(len(windowed_data.index))) # old version
            timed = True if resample else False
            windowed_data = aggregate_in_windows(windowed_data[selected].copy(deep=True), selected, aggregation_length,
                                                 timed, resample, new_features)
            selected = windowed_data.columns.values
            num_of_features = len(selected)
        # and add the new trace
        ints = False if (aggregation and 'encoding' not in old_selected) or 'duration' in old_selected else True
        if windowed_data.shape[0] != 0:  # for the resampling case
            traces += [convert2flexfringe_format(windowed_data[selected], ints)]
        # store also the starting and the ending index of the current time window
        if windowed_data.shape[0] != 0:  # for the resampling case
            traces_indices += [windowed_data.index.tolist()]

    # evaluate correctness of the process
    if not all(list(assertion_dict.values())):
        print('There are flows missed in the current high level window -- Check again the implementation!!!')
        print([k for k, v in assertion_dict.items() if not v], file=sys.stderr)

    else:
        print('All flows correctly converted to traces in the current high level window!!!')

    return traces, traces_indices, num_of_features


def extract_traces(data, out_filepath, selected, alphabet_size, timed=True, dynamic=True, aggregation=False,
                   resample=False, new_features=True):
    """
    Function for extracting traces from the given dataset by first applying a high-level filtering to find windows of
    significant time difference between them to be processed separately by the extract_traces_from_window function. The
    extracted traces are saved in out_filepath.
    :param data: the input dataframe
    :param out_filepath: the relative path of the output traces' file
    :param selected: the features to be used
    :param alphabet_size: the size of the alphabet (positive only if discretized input is given -> 'encoding')
    :param timed: boolean flag about the use of timed windows
    :param dynamic: boolean flag about the use of dynamically changing windows
    :param aggregation: the aggregation flag - if set to True, then aggregation windows are created
    :param resample: the resampling flag - if set to True, then resampling is used in the aggregation windows
    :param new_features: boolean flag specifying if new features should be added to the existing ones
    :return: creates and stores the traces' file extracted from the input dataframe
    """
    traces_indices = []
    traces = []
    progress_list = []
    starting_index = 0
    num_of_features = len(selected)
    # timed version
    if timed:
        medians = data['date'].sort_values().diff().dt.total_seconds()
        high_level_window_indices = medians[medians > 1800].index.tolist()
        if len(high_level_window_indices) != 0:
            print(str(len(high_level_window_indices)) + ' of high level windows identified!!')
            for i in range(len(high_level_window_indices) + 1):
                index = data.shape[0] if i == len(high_level_window_indices) else high_level_window_indices[i]
                windowed_data = data[starting_index:index].copy(deep=True)
                window, stride = set_windowing_vars(windowed_data)
                # special handle in case a zero length window has been returned
                if window.total_seconds() == 0:
                    window = pd.to_timedelta('25ms')
                    stride = pd.to_timedelta('5ms')
                min_trace_len = int(max(windowed_data.shape[0] / 10000, 10))
                max_trace_len = int(max(windowed_data.shape[0] / 100, 1500))
                if windowed_data.shape[0] < min_trace_len:
                    min_trace_len = windowed_data.shape[0]
                if max_trace_len > 1500:
                    max_trace_len = 1500
                new_traces, new_indices, num_of_features = extract_traces_from_window(windowed_data, selected, window,
                                                                                      stride,
                                                                                      (min_trace_len, max_trace_len),
                                                                                      data.shape[0], progress_list,
                                                                                      dynamic=dynamic,
                                                                                      aggregation=aggregation,
                                                                                      resample=resample,
                                                                                      new_features=new_features)
                traces += new_traces
                traces_indices += new_indices
                starting_index = index
        else:
            print('All the dataset is taken into account!!')
            window, stride = set_windowing_vars(data)
            # special handle in case a zero length window has been returned
            if window.total_seconds() == 0:
                window = pd.to_timedelta('25ms')
                stride = pd.to_timedelta('5ms')
            min_trace_len = int(max(data.shape[0] / 10000, 10))
            max_trace_len = int(max(data.shape[0] / 100, 1500))
            if data.shape[0] < min_trace_len:
                min_trace_len = data.shape[0]
            if max_trace_len > 1500:
                max_trace_len = 1500
            new_traces, new_indices, num_of_features = extract_traces_from_window(data, selected, window, stride,
                                                                                  (min_trace_len, max_trace_len),
                                                                                  data.shape[0], progress_list,
                                                                                  dynamic=dynamic,
                                                                                  aggregation=aggregation,
                                                                                  resample=resample,
                                                                                  new_features=new_features)
            traces += new_traces
            traces_indices += new_indices
    # flow version
    else:
        # set the window and stride
        window = 100
        stride = int(window/5)
        # obtain the indices residing in the processed data
        data_indices = data.index.tolist()
        # create a dict for testing if all the flows have been included in the traces
        assertion_dict = dict(zip(data_indices, len(data_indices) * [False]))
        # keep the maximum index
        max_ind = data_indices[-1]
        # keep the original features
        old_selected = deepcopy(selected)
        while starting_index < len(data_indices):
            # retrieve the flows of the current window
            windowed_data = data[starting_index: min(starting_index + window, len(data_indices))]
            # insert the indices of the current trace to the assertion dictionary
            assertion_dict.update(zip(windowed_data.index.tolist(), len(windowed_data.index.tolist()) * [True]))
            # update the maximum seen index (for progress visualization) purposes
            cnt = windowed_data.index.tolist()[-1]
            # create aggregated features if needed (currently with a hard-coded window length)
            if aggregation:
                aggregation_length = 5
                windowed_data = aggregate_static(windowed_data[selected].copy(deep=True), selected, aggregation_length,
                                                 new_features)
                selected = windowed_data.columns.values
                num_of_features = len(selected)
            # extract the trace of this window and add it to the traces' list
            ints = False if aggregation or 'duration' in old_selected else True
            traces += [convert2flexfringe_format(windowed_data[selected], ints)]
            # store also the flow indices of the current time window
            traces_indices += [windowed_data.index.tolist()]
            # retrieve the original features
            selected = deepcopy(old_selected)
            # increase the starting index of the window
            starting_index += stride
            # show progress
            prog = int((cnt / max_ind) * 100)
            if prog // 10 != 0 and prog // 10 not in progress_list:
                progress_list += [prog // 10]
                print('More than ' + str((prog // 10) * 10) + '% of the data processed...')
        # evaluate correctness of the process
        if not all(list(assertion_dict.values())):
            print('There are flows missed in the current high level window -- Check again the implementation!!!')
            print([k for k, v in assertion_dict.items() if not v], file=sys.stderr)

    print('Finished with rolling windows!!!')
    print('Starting writing traces to file...')
    # create the traces' file in the needed format
    f = open(out_filepath, "w")
    if alphabet_size > 0:
        # if symbols are used
        f.write(str(len(traces)) + ' ' + str(alphabet_size) + '\n')
        for trace in traces:
            f.write('1 ' + str(len(trace)) + ' ' + ' '.join(trace) + '\n')
    else:
        # if the multivariate version is used
        f.write(str(len(traces)) + ' ' + '100:' + str(num_of_features) + '\n')
        for trace in traces:
            f.write('1 ' + str(len(trace)) + ' 0:' + ' 0:'.join(trace) + '\n')
    f.close()
    # save also the indices of each trace
    indices_filepath = '.'.join(out_filepath.split('.')[:-1]) + '_indices.pkl'
    with open(indices_filepath, 'wb') as f:
        pickle.dump(traces_indices, f)
    print('Traces written successfully to file!!!')


def parse_dot(dot_path):
    """
    Function for parsing dot files describing multivariate models produced through FlexFringe
    :param dot_path: the path to the dot file
    :return: the parsed model
    """
    with open(dot_path, "r") as f:
        dot_string = f.read()
    # initialize the model
    model = Model()
    # regular expression for parsing a state as well as its contained info
    state_regex = r"(?P<src_state>\d+) \[\s*label=\"(?P<state_info>(.+))\".*\];$"
    # regular expression for parsing a state's contained info
    info_regex = r"((?P<identifier>(fin|symb|attr)+\(\d+\)):\[*(?P<values>(\d|,)+)\]*)+"
    # regular expression for parsing the transitions of a state, as well as the firing conditions
    transition_regex = r"(?P<src_state>.+) -> (?P<dst_state>\d+)( \[label=\"(?P<transition_cond>(.+))\".*\])*;$"
    # regular expression for parsing the conditions firing a transition (it can be used also in case a feature's split
    # is on a floating value or on scientific notation
    cond_regex = r"((?P<sym>\d+) (?P<ineq_symbol>(<|>|<=|>=|==)) (?P<boundary>\d+(\.\d+(e\+\d+){0,1}){0,1}))+"
    # boolean flag used for adding states in the model
    cached = False
    for line in re.split(r'\n\t+|\n}', dot_string):  # split the dot file in meaningful lines
        state_matcher = re.match(state_regex, line, re.DOTALL)
        # if a new state is found while parsing
        if state_matcher is not None:
            # check if there is a state pending and, if there is, update the model
            if cached:
                model.add_node(ModelNode(src_state, attributes, fin, tot, dst_nodes, cond_dict))
            src_state = state_matcher.group("src_state")  # find the state number
            # and initialize all its parameters
            attributes = {}
            fin = 0
            tot = 0
            dst_nodes = set()
            cond_dict = {}
            # find the state information while parsing
            state_info = state_matcher.group("state_info")
            if state_info != 'root':
                # if the state is not the root one retrieve the state's information
                for info_matcher in re.finditer(info_regex, state_info):
                    identifier = info_matcher.group("identifier")
                    id_values = info_matcher.group("values")
                    if 'fin' in identifier:  # the number of times the state was final
                        fin = int(id_values)
                    elif 'symb' in identifier:  # the number of times the state was visited
                        tot = fin + int(id_values[:-1])
                    else:
                        # the quantiles of each attribute
                        attributes[re.findall(r'\d+', identifier)[0]] = list(map(float, id_values.split(',')[:-1]))
            else:
                # otherwise set the state's name to root (because for some reason 3 labels are used for the root node)
                src_state = 'root'

        transition_matcher = re.match(transition_regex, line, re.DOTALL)
        # if a transition is identified while parsing (should be in the premises of the previously identified state)
        if transition_matcher is not None:
            src_state_1 = transition_matcher.group("src_state")  # the source state number
            if src_state_1 == 'I':  # again the same problem as above
                src_state_1 = 'root'
            # just consistency check that we are still parsing the same state as before
            if src_state != src_state_1:
                print('Something went wrong - Different source states in a state!!!')
                return -1
            # identify the destination state and add it to the list of destinations
            dst_state = transition_matcher.group("dst_state")
            dst_nodes.add(dst_state)  # should exist given that transitions come after the identification of a new state
            # check for the transitions' conditions only if the current state is not the root
            if src_state_1 != 'root':
                # find the transition's conditions while parsing
                transition_conditions = transition_matcher.group("transition_cond")
                conditions_to_be_added = []
                for condition_matcher in re.finditer(cond_regex, transition_conditions):
                    attribute = condition_matcher.group("sym")  # the attribute contained in the condition
                    inequality_symbol = condition_matcher.group("ineq_symbol")  # the inequality symbol
                    boundary = condition_matcher.group("boundary")  # the numeric limit in the condition
                    # and set the conditions to be added in the conditions' dictionary
                    conditions_to_be_added += [(attribute, inequality_symbol, float(boundary))]

                # the condition dictonary should be initialized from the state identification stage
                if dst_state not in cond_dict.keys():
                    cond_dict[dst_state] = []
                cond_dict[dst_state] += [conditions_to_be_added]
            # set the cached flag to True after the first state is fully identified
            cached = True

    # one more node addition for the last state in the file to be added
    model.add_node(ModelNode(src_state, attributes, fin, tot, dst_nodes, cond_dict))
    return model


def parse_symbolic_dot(dot_path):
    """
    Function for parsing a dot file describing sequential models with a symbolic alphabet
    :param dot_path: the path to the dot file
    :return: the parsed model
    """
    with open(dot_path, "r") as f:
        dot_string = f.read()
    model = {}
    SYMLST_REGEX = "((?P<sym>\d+):(?P<occ>\d+))+"
    TRAN_REGEX = "(?P<sst>.+) -> (?P<dst>.+) \[label=\"(?P<slst>(.+))\"[ style=dotted]*  \];$"
    for line in dot_string.split('\n'):
        matcher = re.match(TRAN_REGEX, line.strip())
        if matcher is not None:
            sstate = matcher.group("sst")
            dstate = matcher.group("dst")
            symlist = matcher.group("slst")
            if sstate not in model:
                model[sstate] = {}
            for smatcher in re.finditer(SYMLST_REGEX, symlist):
                symbol = smatcher.group("sym")
                occurrences = int(smatcher.group("occ"))
                model[sstate][symbol] = (dstate, occurrences)

    # normalizing occurrence counts to probabilities | add Laplace smoothing for unseen symbols
    for state in model:
        # find the sum of all occurrences and add 1 for smoothing
        ssum = sum(map(lambda x: x[1]+1, model[state].values()))
        for symbol in model[state]:
            dstate, occurrences = model[state][symbol]
            model[state][symbol] = (dstate, occurrences, (occurrences + 1) / ssum)
        # finally add an unseen symbol
        model[state]['unseen'] = (state, 0, 1 / ssum)
    return model


def traces2list(traces_path):
    """
    Function for converting the multivariate trace file into a list of traces ready for further processing
    :param traces_path: the filepath of the traces
    :return: the list of the traces as a list of lists (each trace) of lists (each record in each trace)
    """
    traces = []
    with open(traces_path, "r") as fp:
        # skip first line
        line = fp.readline()
        while line:
            line = fp.readline()
            if line != '':
                # split lines by spaces
                tokens = line.split()
                # gather the records of each trace and keep only the record values and map them to float
                traces += [[list(map(float, t.split(':')[1].split(','))) for t in tokens[2:]]]
    return traces


def symbolic_traces2list(traces_path):
    """
    Function for converting the symbolic trace file into a list of traces ready for further processing
    :param traces_path: the filepath of the traces
    :return: the list of the traces as a list of lists (each trace) of lists (each record in each trace)
    """
    traces = []
    with open(traces_path, "r") as fp:
        # skip first line
        line = fp.readline()
        while line:
            line = fp.readline()
            if line != '':
                # split lines by spaces
                tokens = line.split()
                # gather the symbols of each trace
                traces += [tokens[2:]]
    return traces


def run_traces_on_model(traces_path, indices_path, model, attribute_type='train'):
    """
    Function for running a trace file on the provided model and storing the observed attributes' values on it
    :param traces_path: the filepath to the traces' file
    :param indices_path: the filepath to the traces' incices file
    :param model: the given model
    :param attribute_type: the type of the input traces ('train' | 'test')
    :return: the updated model
    """
    traces = traces2list(traces_path)
    with open(indices_path, 'rb') as f:
        traces_indices = pickle.load(f)
    for trace, inds in zip(traces, traces_indices):
        # first fire the transition from root node
        label = model.fire_transition('root', dict())
        for record, ind in zip(trace, inds):
            observed = dict(zip([str(i) for i in range(len(record))], record))
            model.update_attributes(label, observed, attribute_type)
            model.update_indices(label, ind, attribute_type)
            label = model.fire_transition(label, observed)
    return model


def run_traces_on_symbolic_model(traces_path, model, eval_method='acceptance', train_path=None):
    """
    Function for running a symbolic trace file on the provided model and returning the number of accepted and rejected
    traces. The acceptance and rejection of a trace depends on the evaluation method provided
    :param traces_path: the path of the trace file to run on the model
    :param model: the given model
    :param eval_method: the evaluation method to be used for the set of symbolic sequences given as input ('acceptance'
    | 'error')
    :param train_path: the path of the trace file used to create the model (provided in the error-based evaluation
    method)
    :return: the ratio of accepted traces (in case the acceptance method is used) or the normalised sum of pattern
    errors between the training and testing traces (in case the error-based method is used)
    """
    traces = symbolic_traces2list(traces_path)
    if eval_method == 'acceptance':
        accepted = 0
        rejected = 0
        for trace in traces:
            state = '0'
            prob = 1
            for event in trace:
                # if the symbol is found at the current state, the new state is retrieved
                if event in model[state].keys():
                    state = model[state][event][0]
                else:
                    # the word is rejected
                    prob = 0
                    break
            if prob == 0:   # count a rejected word
                rejected += 1
            else:           # count an accepted word
                accepted += 1
        return accepted / (accepted + rejected)
    else:
        err = 0
        # retrieve the traces that created the model
        train_traces = symbolic_traces2list(train_path)
        # initialize the counters for the total number od pattern occurrences in each type of traces
        total_train = 0
        total_test = 0
        # initialize the dictionaries of the patterns in each type of traces
        train_dict = {}
        test_dict = {}
        for state in model.keys():
            train_dict[state] = defaultdict(int)
            test_dict[state] = defaultdict(int)
        # set the patterns occurrences by running the tracets of each type
        for ind, curr_traces in enumerate([train_traces, traces]):
            for trace in curr_traces:
                state = '0'
                for event in trace:
                    # if the symbol is found at the current state, the new state is retrieved
                    if event in model[state].keys():
                        state = model[state][event][0]
                        if ind == 0:
                            train_dict[state][event] += 1
                            total_train += 1
                        else:
                            test_dict[state][event] += 1
                            total_test += 1
                    else:
                        # the word is rejected
                        break
        # finally find the normalized sum of absolute differences between the normalized counts of each pattern
        total_errors = 0
        for state in model.keys():
            for symbol in set(train_dict[state].keys()).union(set(test_dict[state].keys())):
                err += abs(train_dict[state][symbol] / total_train - (0 if not total_test else test_dict[state][symbol] / total_test))
                total_errors += 1
        return err / total_errors


def dict2list(d):
    """
    Function for converting a dictionary to list by using the keys of the dictionary as the indices of the list and its
    values as the values of the list.
    :param d: the input dictionary
    :return: the output list
    """
    l = [0] * len(d.keys())
    for k in d.keys():
        l[k] = d[k]
    return l
