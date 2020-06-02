#!/usr/bin/python

import subprocess
import os
import glob
import graphviz
import helper
import initial_clustering
import pandas as pd
import re
from copy import deepcopy
import pickle
from random import sample

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

    # rename the output file to an indicating name
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
    flag = 'CTU-bi'

    with_trace = int(input('Is there a trace file (no: 0 | yes: 1)? '))

    if not with_trace:
        # first check if discretization is enabled
        with_discretization = int(
            input('Discretize numeric features (ports, bytes, duration, packets) (no: 0 | yes: 1)? '))
        # set the features to be used in the multivariate modelling
        if flag in ['CTU-bi', 'UNSW', 'CICIDS']:
            selected = [
                # 'src_port'
                'dst_port'
                , 'protocol_num'
                # , 'duration'
                # , 'src_bytes'
                # , 'dst_bytes'
                # , 'date_diff'
                , 'total_bytes'
                , 'bytes_per_packet'
                        ]
        else:
            selected = [
                # 'src_port'
                'dst_port'
                , 'protocol_num'
                # , 'duration'
                , 'orig_ip_bytes'
                , 'resp_ip_bytes'
                # , 'date_diff'
                # , 'orig_bytes_per_packet'
                # , 'resp_bytes_per_packet'
            ]
        old_selected = deepcopy(selected)

        # list of source ips to solely consider
        src_ips = ['147.32.84.170', '147.32.80.9']

        host_level = int(input('Select the type of modelling to be conducted (connection level: 0 | host level: 1 | '
                               'mixed-adaptive: 2): '))
        if not host_level:
            analysis_type = 'connection_level'
        elif host_level == 1:
            analysis_type = 'host_level'
        else:
            analysis_type = 'mixed_adaptive'
        bidirectional = False

        # set the input filepath
        training_filepath = input('Give the relative path of the dataframe to be used for training: ')

        if flag == 'CTU-bi':
            data = pd.read_pickle(training_filepath + '/binetflow_normal.pkl')
        elif flag == 'IOT':
            data = pd.read_pickle(training_filepath + '/zeek_normal.pkl')
        else:
            data = pd.read_pickle(training_filepath + '/normal.pkl')

        if with_discretization:
            # first find the discretization limits for each feature
            discretization_dict = helper.find_discretization_clusters(data, [sel for sel in old_selected if 'num'
                                                                             not in sel])
            # and store them in a pickle in the training scenario's directory
            discretization_filepath = training_filepath + '/discretization_limits.pkl'
            with open(discretization_filepath, 'wb') as f:
                pickle.dump(discretization_dict, f)
            # then for each feature discretize its values and add the new features in the dataframe
            for feature in discretization_dict.keys():
                data[feature + '_num'] = data[feature].apply(helper.find_percentile,
                                                             args=(discretization_dict[feature],))
                selected.remove(feature)
                selected += [feature + '_num']
            old_selected = deepcopy(selected)

        # select instances according to a different level of analysis
        if analysis_type == 'host_level':
            # in host level analysis only host with significant number of flows are considered -> this could be
            # problematic in hosts with intractably large number of flows
            datatype = 'non-regular' if flag == 'IOT' else 'regular'
            instances = helper.select_hosts(data, 40, bidirectional=bidirectional, datatype=datatype).values.tolist()
            print('Number of hosts to be processed: ' + str(len(instances)))
        elif analysis_type == 'mixed_adaptive':
            # TODO: remove this part since it is not robust
            # firstly create lists to include the different abstraction level data
            instances = []
            sets_of_data = []
            # the mixed analysis type according to which 3 levels of abstraction are examined
            # In the first abstraction level, connections are split to those with a significant amount of flows for
            # modelling and those with a very small number of flows. The point is to use the first ones regularly and
            # further preprocess the second ones
            big_data, small_data = initial_clustering.separate_by_connections(data, 100)
            big_instances = big_data.groupby(['src_ip', 'dst_ip']).size().reset_index().values.tolist()
            print('Number of long connections to be processed: ' + str(len(big_instances)))
            small_instances = small_data.groupby(['src_ip', 'dst_ip']).size().reset_index().values.tolist()
            print('Number of small connections to be processed: ' + str(len(small_instances)))
            sets_of_data.append(big_data)
            instances += big_instances

            # In the second abstraction level, the short connections are split to those which demonstrate a small number
            # of unique ports, meaning a considerable number of flows for each port, and those with a high number of
            # unique ports, meaning a small number of flows per port. The point is to use the flows dedicated to
            # specific ports and learn models from them
            small_data_few_ports, small_data_many_ports = initial_clustering.separate_by_ports(small_data, 100)
            few_ports_instances = small_data_few_ports['dst_port'].unique()
            print('Number of unique ports to be processed in small connections with few ports: ' +
                  str(len(few_ports_instances)))
            many_ports_instances = small_data_many_ports['dst_port'].unique()
            print('Number of unique ports to be processed in small connections with many ports: ' +
                  str(len(many_ports_instances)))
            sets_of_data.append(small_data_few_ports)
            instances += few_ports_instances.tolist()

            # In the final level of abstraction, the connections with small number of flows and multiple unique
            # destination ports are considered. These flows are split according to the number of bytes received (zero or
            # non-zero). The point here is to treat the connections of each category as a dense cluster and learn a
            # model for both clusters.
            zero_dst_instances, non_zero_dst_instances = initial_clustering.separate_by_dst_bytes(small_data_many_ports)
            print('Number of flows for small connections with many ports and zero received bytes: ' +
                  str(len(zero_dst_instances)))
            print('Number of flows for small connections with many ports and some received bytes: ' +
                  str(len(non_zero_dst_instances)))
            instances.append('zero_received_bytes')
            sets_of_data.append(zero_dst_instances)
            instances.append('some_received_bytes')
            sets_of_data.append(non_zero_dst_instances)
        else:
            # in connection level analysis only connections with significant number of flows are considered ->
            # this could be problematic in connections with intractably large number of flows (not so probable to happen
            # benign scenarios)
            datatype = 'non-regular' if flag == 'IOT' else 'regular'
            if src_ips:
                data = data.loc[data['src_ip'].isin(src_ips)].reset_index(drop=True)
            instances = helper.select_connections(data, 40, bidirectional=bidirectional, datatype=datatype).values.tolist()
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
                print('The number of flows for this host are: ' + str(instance_data.shape[0]))
            elif analysis_type == 'mixed_adaptive':
                # in case the instance is a list then we are dealing with the connection level
                if isinstance(instances[j], list):
                    instance_name = instances[j][0] + '-' + instances[j][1]
                    print('Extracting traces for connection ' + instance_name)
                    data = sets_of_data[ind]
                    instance_data = data.loc[(data['src_ip'] == instances[j][0]) & (data['dst_ip'] == instances[j][1])]. \
                        sort_values(by='date').reset_index(drop=True)
                    print('The number of flows for this connection are: ' + str(instance_data.shape[0]))
                    prev = instances[j]
                # in case the instance is a string we are dealing with the received bytes data
                elif isinstance(instances[j], str):
                    # check if it is the first time we see a string instance
                    if not isinstance(prev, str):
                        # first deal with the case of the flows with zero received bytes
                        ind += 1
                        instance_name = instances[j]
                        print('Extracting traces for connections with ' + instance_name)
                        data = sets_of_data[ind]
                        if data.shape[0] > 20000:
                            # in this case where there is no sense of group clustering will be used if the size of the
                            # data are excessive
                            # -1 is used so that the received bytes are not included in the clustering procedure
                            clusters = initial_clustering.apply_kmeans(data[selected[:-1]].values)
                            for i in range(clusters.cluster_centers_.shape[0]):
                                instance_data = data[clusters.labels_ == i]
                                instances.append(instance_name + 'cluster_' + str(i))
                                sets_of_data.append(instance_data)
                        else:
                            instances.append(instance_name)
                            sets_of_data.append(data)
                        prev = instances[j]
                        # then deal with the case of the flows with some received bytes
                        j += 1
                        ind += 1
                        instance_name = instances[j]
                        print('Extracting traces for connections with ' + instance_name)
                        data = sets_of_data[ind]
                        if data.shape[0] > 20000:
                            # in this case where there is no sense of group clustering will be used if the size of the
                            # data are excessive
                            clusters = initial_clustering.apply_kmeans(data[selected].values)
                            for i in range(clusters.cluster_centers_.shape[0]):
                                instance_data = data[clusters.labels_ == i]
                                instances.append(instance_name + 'cluster_' + str(i))
                                sets_of_data.append(instance_data)
                        else:
                            instances.append(instance_name)
                            sets_of_data.append(data)
                        prev = instances[j]
                        j += 1
                        # we continue so that no data is falsely processed in the case that clusters have been created
                        continue
                    # in case we have seen string instances before we have to process the data
                    else:
                        ind += 1
                        instance_name = instances[j]
                        instance_data = sets_of_data[ind]
                        print('The number of flows for this clustered partition are: ' + str(instance_data.shape[0]))
                        prev = instances[j]
                # otherwise we are dealing with the port level
                else:
                    # if the previous instance was a list then an increase in the index of the set of data to process
                    # should occur
                    if isinstance(prev, list):
                        ind += 1
                    instance_name = str(instances[j])
                    print('Extracting traces for connections with destination port ' + instance_name)
                    instance_name = 'port_' + instance_name
                    data = sets_of_data[ind]
                    instance_data = data.loc[data['dst_port'] == instances[j]].sort_values(by='date').reset_index(drop=True)
                    print('The number of flows for this destination port are: ' + str(instance_data.shape[0]))
                    prev = instances[j]
            else:
                instance_name = instances[j][0] + '-' + instances[j][1]
                print('Extracting traces for connection ' + instance_name)
                instance_data = data.loc[(data['src_ip'] == instances[j][0]) & (data['dst_ip'] == instances[j][1])].\
                    sort_values(by='date').reset_index(drop=True)
                print('The number of flows for this connection are: ' + str(instance_data.shape[0]))

            # if the number of flows is excessive then sample from these flows to a feasible extent
            if instance_data.shape[0] > 20000:
                # instance_data = instance_data.iloc[sample(range(instance_data.shape[0]), k=20000)].\
                #     sort_values(by='date').reset_index(drop=True)
                instance_data = instance_data.iloc[:20000]

            # create a column with the time difference between consecutive flows
            instance_data['date_diff'] = instance_data['date'].sort_values().diff().astype('timedelta64[ms]') * 0.001
            instance_data['date_diff'].fillna(0, inplace=True)

            # create column with the ratio of bytes to packets
            if flag == 'CTU-bi':
                instance_data['total_bytes'] = instance_data['src_bytes'] + instance_data['dst_bytes']
                instance_data['bytes_per_packet'] = instance_data['total_bytes'] / instance_data['packets']
                instance_data['bytes_per_packet'].fillna(0, inplace=True)
            elif flag == 'UNSW':
                instance_data['total_bytes'] = instance_data['src_bytes'] + instance_data['dst_bytes']
                instance_data['bytes_per_packet'] = instance_data['total_bytes'] / (instance_data['src_packets'] +
                                                                                    instance_data['dst_packets'])
                instance_data['bytes_per_packet'].fillna(0, inplace=True)
            elif flag == 'CICIDS':
                instance_data['total_bytes'] = instance_data['src_bytes'] + instance_data['dst_bytes']
                instance_data['bytes_per_packet'] = instance_data['total_bytes'] / (instance_data['total_fwd_packets'] +
                                                                                    instance_data['total_bwd_packets'])
                instance_data['bytes_per_packet'].fillna(0, inplace=True)
            else:
                instance_data['total_bytes'] = instance_data['orig_ip_bytes'] + instance_data['resp_ip_bytes']
                instance_data['bytes_per_packet'] = instance_data['total_bytes'] / (instance_data['orig_packets'] +
                                                                                    instance_data['resp_packets'])
                instance_data['bytes_per_packet'].fillna(0, inplace=True)

            # first ask if new features are to be added
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
                                  instance_name + '-traces' + ('_bdr' if bidirectional else '') + '.txt'
                aggregation = False
                resample = False
            else:
                resample = False if aggregation == 1 else True
                aggregation = True
                if resample:
                    traces_filepath = '/'.join(training_filepath.split('/')[0:2]) + '/training/' + analysis_type \
                                      + '/' + '_'.join(old_selected) + '/' + training_filepath.split('/')[2] + '-' \
                                      + instance_name + '-traces_resampled' + ('' if new_features else '_reduced') \
                                      + ('_bdr' if bidirectional else '') + '.txt'
                else:
                    traces_filepath = '/'.join(training_filepath.split('/')[0:2]) + '/training/' + analysis_type \
                                      + '/' + '_'.join(old_selected) + '/' + training_filepath.split('/')[2] + '-' \
                                      + instance_name + '-traces_aggregated' + ('' if new_features else '_reduced') \
                                      + ('_bdr' if bidirectional else '') + '.txt'
                # add also the destination ip in case of aggregation
                if analysis_type == 'host_level':
                    selected += ['dst_ip'] if not resample else ['dst_ip', 'date']
                else:
                    if resample:
                        selected += ['date']

            # create the directory if it does not exist
            os.makedirs(os.path.dirname(traces_filepath), exist_ok=True)

            # and extract the traces
            helper.extract_traces(instance_data, traces_filepath, selected, dynamic=True, aggregation=aggregation,
                                  resample=resample, new_features=bool(new_features))

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
