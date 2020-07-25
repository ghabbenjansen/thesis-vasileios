#!/usr/bin/python

import subprocess
import os
import glob
import graphviz
import helper
import pandas as pd
import re
from copy import deepcopy
import pickle

# flag for specifying which version of flexfringe shall be used. In case the symbolic approach is used the master branch
# is used, otherwise the mutlivariate branch is used
ENCODED = False
if ENCODED:
    # local filepath where the code of the master branch resides
    filepath = '/Users/vserentellos/Desktop/dfasat/'
else:
    # local filepath where the code of the multivariate branch resides
    filepath = '/Users/vserentellos/Documents/dfasat/'


def flexfringe(*args, **kwargs):
    """
    Wrapper function for running flexfringe in Python
    :param args: the input arguments for flexfringe
    :param kwargs: the keywords arguments (flags with their values) for flexfringe
    :return: the opened dot file of the output of flexfringe, as well as the filepath that it is stored
    """
    command = ["--help"]

    # collect the keyword arguments
    if len(kwargs) > 1:
        command = ["-" + key + "=" + kwargs[key] for key in kwargs]

    # run flexfringe with the given arguments
    print("%s" % subprocess.run([filepath + "flexfringe", ] + command + [args[0]], stdout=subprocess.PIPE, check=True)
          .stdout.decode())

    # remove unnecessary files
    # first the initial dfa file
    for f in glob.glob("outputs/init*"):
        os.remove(f)
    # and then then pre and post refs dot files that flexfringe creates
    for f in glob.glob("*.dot"):
        os.remove(f)

    # rename the output file to an indicating name according to the version of flexfringe
    if ENCODED:
        old_file = os.path.join("outputs", "final.dot")
    else:
        old_file = os.path.join("outputs", "final.json")
    # extract the features that have been used
    features = args[0].split('/')[-2]
    dataset_name = args[0].split('/')[1]
    analysis_level = args[0].split('/')[-3]
    extension = re.search('(.+?)-traces', args[0].split('/')[-1]).group(1)
    # add this naming in case aggregation windows have been used
    if 'aggregated' in args[0]:
        extension += ('_aggregated' + ('_reduced' if 'reduced' in args[0] else ''))
    if 'resampled' in args[0]:
        extension += ('_resampled' + ('_reduced' if 'reduced' in args[0] else ''))
    if 'static' in args[0]:
        extension += '_static'
    new_file_name = extension + "_dfa.dot"
    new_file = os.path.join("outputs/" + dataset_name + '/' + analysis_level + '/' + features, new_file_name)

    # create the directory if it does not exist and rename the created dot file
    os.makedirs(os.path.dirname(new_file), exist_ok=True)
    os.rename(old_file, new_file)

    # and open the output dot file
    try:
        with open("outputs/" + dataset_name + '/' + analysis_level + '/' + features + "/" + new_file_name) as fh:
            return fh.read(), "outputs/" + dataset_name + '/' + analysis_level + '/' + features + "/" + new_file_name
    except FileNotFoundError:
        pass

    return "No output file was generated."


def show(data, filepath):
    """
    Function for plotting a dot file, created after a run of flexfringe, through graphviz
    :param data: the content of the dot file
    :param filepath: the filepath to be used when plotting the model
    :return: plots the created automaton provided in data
    """
    if data == "":
        pass
    else:
        # first extract the directory and filename from the filepath
        directory = '/'.join(filepath.split('/')[0:-1])
        filename = filepath.split('/')[-1]
        filename = '.'.join(filename.split('.')[:-1])  # and then remove the '.dot' ending
        # and then create the dfa plot
        g = graphviz.Source(data, filename=filename, directory=directory, format="png")
        g.render(filename=filename, directory=directory, view=True, cleanup=True)


if __name__ == '__main__':
    # first set the flag of the type of dataset to be used
    # flag should be one of 'CTU-bi' | 'UNSW' | 'CICIDS'
    flag = 'CTU-bi'

    with_trace = int(input('Is there a trace file (no: 0 | yes: 1)? '))

    if not with_trace:
        # first check if discretization is enabled
        with_discretization = int(
            input('Discretize numeric features (ports, bytes, duration, packets) (no: 0 | yes: 1)? '))
        # set the features to be used in the multivariate modelling
        selected = [
            'src_port'
            , 'dst_port'
            , 'protocol_num'
            , 'duration'
            , 'src_bytes'
            , 'dst_bytes'
            # , 'total_bytes'
            # , 'bytes_per_packet'
                    ]
        old_selected = deepcopy(selected)

        # list of source ips to solely consider
        src_ips = []

        # flag for keeping only major connections #TODO: remove if for final version
        major_flag = False

        # alphabet size in case the discretized version is used
        alphabet_size = -1

        host_level = int(input('Select the type of modelling to be conducted (connection level: 0 | host level: 1): '))
        if not host_level:
            analysis_type = 'connection_level'
        else:
            analysis_type = 'host_level'
        bidirectional = False

        # set the input filepath
        training_filepath = input('Give the relative path of the dataframe to be used for training: ')

        if flag == 'CTU-bi':
            data = pd.read_pickle(training_filepath + '/binetflow_normal.pkl')
        else:
            data = pd.read_pickle(training_filepath + '/normal.pkl')

        # create column with the ratio of bytes to packets
        if flag == 'CTU-bi':
            data['total_bytes'] = data['src_bytes'] + data['dst_bytes']
            data['bytes_per_packet'] = data['total_bytes'] / data['packets']
            data['bytes_per_packet'].fillna(0, inplace=True)
        elif flag == 'UNSW':
            data['total_bytes'] = data['src_bytes'] + data['dst_bytes']
            data['bytes_per_packet'] = data['total_bytes'] / (data['src_packets'] + data['dst_packets'])
            data['bytes_per_packet'].fillna(0, inplace=True)
        else:
            data['total_bytes'] = data['src_bytes'] + data['dst_bytes']
            data['bytes_per_packet'] = data['total_bytes'] / (data['total_fwd_packets'] + data['total_bwd_packets'])
            data['bytes_per_packet'].fillna(0, inplace=True)

        if with_discretization:
            # first find the discretization limits for each feature
            discretization_dict = helper.find_discretization_clusters(data, [sel for sel in old_selected if 'num'
                                                                             not in sel])
            # added part for creating encoding of the data for the baseline discretized version
            # just keep the alphabet size for already discretized features -> for the rest it can be induced from the
            # number of percentiles in the dictionary
            for sel in old_selected:
                if 'num' in sel:
                    discretization_dict[sel] = sorted(data[sel].unique().tolist())
            # and store them in a pickle in the training scenario's directory
            discretization_filepath = training_filepath + '/discretization_limits.pkl'
            with open(discretization_filepath, 'wb') as f:
                pickle.dump(discretization_dict, f)
            # then for each feature discretize its values and add the new features in the dataframe
            alphabet_size = 1
            for feature in discretization_dict.keys():
                alphabet_size *= (len(discretization_dict[feature]) + 1)
                if 'num' not in feature:
                    data[feature + '_num'] = data[feature].apply(helper.find_percentile,
                                                                 args=(discretization_dict[feature],))
                    selected.remove(feature)
                    selected += [feature + '_num']
                else:
                    data[feature] = data[feature].apply(helper.check_existence,
                                                                 args=(discretization_dict[feature],))
            helper.netflow_encoding(data, selected, discretization_dict)
            selected = ['encoding']
            old_selected = deepcopy(selected)

        # select instances according to a different level of analysis
        if analysis_type == 'host_level':
            # in host level analysis only host with significant number of flows are considered
            instances = helper.select_hosts(data, 50, bidirectional=bidirectional).values.tolist()  # 10000
            print('Number of hosts to be processed: ' + str(len(instances)))
        else:
            # in connection level analysis only connections with significant number of flows are considered
            if src_ips:
                # in case only certain IPs should be considered
                data = data.loc[data['src_ip'].isin(src_ips)].reset_index(drop=True)
            instances = helper.select_connections(data, 40, bidirectional=bidirectional).values.tolist()
            print('Number of connections to be processed: ' + str(len(instances)))

        # initialize an empty list to hold the filepaths of the trace files for each host
        traces_filepaths = []
        j = 0
        # variables used only in the case of the mixed analysis
        prev = None
        ind = 0
        # extract the data according to the analysis level
        while j < len(instances):
            if analysis_type == 'host_level':
                instance_name = instances[j][0]
                print('Extracting traces for host ' + instance_name)
                instance_data = data.loc[data['src_ip'] == instances[j][0]].sort_values(by='date').reset_index(drop=True)
                # check if only the major connection should be kept -> set only for CICIDS
                if major_flag:
                    instance_data = helper.keep_only_major_connection(instance_data)
                print('The number of flows for this host are: ' + str(instance_data.shape[0]))
            else:
                instance_name = instances[j][0] + '-' + instances[j][1]
                print('Extracting traces for connection ' + instance_name)
                instance_data = data.loc[(data['src_ip'] == instances[j][0]) & (data['dst_ip'] == instances[j][1])].\
                    sort_values(by='date').reset_index(drop=True)
                print('The number of flows for this connection are: ' + str(instance_data.shape[0]))

            # if the number of flows is excessive then sample from these flows to a feasible extent
            if instance_data.shape[0] > 20000:
                instance_data = instance_data.iloc[:20000]

            # first ask about the nature of the windowing technique
            timed = int(input('What type of window to use (0: non-timed | 1: static-timed | 2: dynamic-timed)? '))
            dynamic = True if timed == 2 else False
            timed = bool(timed)
            # if static windows are used add it to the naming of the tracefile
            timed_name = ''
            if not timed:
                timed_name = '_static'
            elif not dynamic:
                timed_name = '_static_timed'

            # seconldy ask if new features are to be added
            new_features = int(input('Are there any new features to be added (no: 0 | yes: 1)? '))

            # extract the traces and save them in the traces' filepath
            aggregation = int(input('Do you want to use aggregation windows (no: 0 | yes-rolling: 1 | yes-resample:'
                                    ' 2 )? '))

            # set the traces output filepath depending on the aggregation value
            # if aggregation has been set to 1 then proper naming is conducted in the extract_traces function of the
            # helper.py file
            if not aggregation:
                traces_filepath = '/'.join(training_filepath.split('/')[0:2]) + '/training/' + analysis_type + '/' \
                                  + '_'.join(old_selected) + '/' + training_filepath.split('/')[2] + '-' + \
                                  instance_name + '-traces' + ('_bdr' if bidirectional else '') + timed_name + '.txt'
                aggregation = False
                resample = False
            else:
                resample = False if aggregation == 1 else True
                aggregation = True
                if resample:
                    traces_filepath = '/'.join(training_filepath.split('/')[0:2]) + '/training/' + analysis_type \
                                      + '/' + '_'.join(old_selected) + '/' + training_filepath.split('/')[2] + '-' \
                                      + instance_name + '-traces_resampled' + ('' if new_features else '_reduced') \
                                      + ('_bdr' if bidirectional else '') + timed_name + '.txt'
                else:
                    traces_filepath = '/'.join(training_filepath.split('/')[0:2]) + '/training/' + analysis_type \
                                      + '/' + '_'.join(old_selected) + '/' + training_filepath.split('/')[2] + '-' \
                                      + instance_name + '-traces_aggregated' + ('' if new_features else '_reduced') \
                                      + ('_bdr' if bidirectional else '') + timed_name + '.txt'
                # add also the destination ip in case of aggregation
                if analysis_type == 'host_level':
                    selected += ['dst_ip'] if not resample else ['dst_ip', 'date']
                else:
                    if resample:
                        selected += ['date']

            # create the directory if it does not exist
            os.makedirs(os.path.dirname(traces_filepath), exist_ok=True)

            # and extract the traces
            helper.extract_traces(instance_data, traces_filepath, selected, alphabet_size, timed=timed, dynamic=dynamic,
                                  aggregation=aggregation, resample=resample, new_features=bool(new_features))

            # add the trace filepath of each host's traces to the list
            traces_filepaths += [traces_filepath]
            # and reset the selected features
            selected = deepcopy(old_selected)
            j += 1
    else:
        # in case the traces' filepath already exists, provide it (in this case only one path - NOT a list
        traces_filepaths = [input('Give the path to the input file for flexfringe: ')]

    # create a model for each host
    for traces_filepath in traces_filepaths:
        # and set the flags for flexfringe
        extra_args = input('Give any flag arguments for flexfinge in a key value way separated by comma in between '
                           'e.g. key1:value1,ke2:value2,...: ').split(',')

        # run flexfringe to produce the automaton and plot it
        modelled_data, storing_path = flexfringe(traces_filepath, **dict([arg.split(':') for arg in extra_args]))
        show(modelled_data, storing_path)
