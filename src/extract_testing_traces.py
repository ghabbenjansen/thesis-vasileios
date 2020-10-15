#!/usr/bin/python

import os
from src import helper
import pandas as pd
from copy import deepcopy
import pickle

if __name__ == '__main__':
    # first set the flag of the type of dataset to be used
    # flag should be one of 'CTU-bi' | 'UNSW' | 'CICIDS'
    while True:
        flag = input("Enter the desired flag (CTU-bi | CICIDS | UNSW): ")
        if flag in ['CTU-bi', 'CICIDS', 'UNSW']:
            break

    # Check if discretization is enabled
    with_discretization = int(input('Discretize numeric features (ports, bytes, duration, packets) (no: 0 | yes: 1)? '))

    # set the flag for selecting only network entities with a sufficient number of flows
    select_major = True

    # flag for modifications in trace extraction for state-of-the-art experiments
    sota = True

    # set the features to be used in the multivariate modelling
    selected = [
        'src_port'
        , 'dst_port'
        , 'protocol_num'
        # , 'duration'
        , 'src_bytes'
        , 'dst_bytes'
                ]
    old_selected = deepcopy(selected)

    host_level = int(input('Select the type of modelling to be conducted (connection level: 0 | host level: 1): '))
    analysis_type = 'host_level' if host_level else 'connection_level'
    bidirectional = False

    # alphabet size in case the discretized version is used
    alphabet_size = -1

    # set the input filepath of the dataframes' directory
    testing_filepath = input('Give the relative path of the dataset to be used for testing: ')
    if flag == 'CTU-bi':
        normal = pd.read_pickle(testing_filepath + '/binetflow_normal.pkl') if not sota else \
            pd.read_pickle(testing_filepath + '/binetflow_normal_sota.pkl')
        anomalous = pd.read_pickle(testing_filepath + '/binetflow_anomalous.pkl')
    else:
        normal = pd.read_pickle(testing_filepath + '/normal.pkl')
        anomalous = pd.read_pickle(testing_filepath + '/anomalous.pkl')
    data = pd.concat([normal, anomalous], ignore_index=True).reset_index(drop=True)

    if with_discretization:
        # first retrieve the discretization limits to be used for each feature
        discretization_filepath = input('Provide the filepath of the discretization limits: ')
        with open(discretization_filepath, 'rb') as f:
            discretization_dict = pickle.load(f)

        alphabet_size = 1
        # then apply discretization to the appropriate features
        for feature in discretization_dict.keys():
            alphabet_size *= (len(discretization_dict[feature]) + 1)
            if 'num' not in feature:
                data[feature + '_num'] = data[feature].apply(helper.find_percentile,
                                                             args=(discretization_dict[feature],))
                selected.remove(feature)
                selected += [feature + '_num']
            else:
                data[feature] = data[feature].apply(helper.check_existence, args=(discretization_dict[feature],))
        helper.netflow_encoding(data, selected, discretization_dict)
        selected = ['encoding']
        old_selected = deepcopy(selected)

    # for testing keep only hosts or connection that have a sufficient number of flows so that enough information is
    # available
    if host_level:
        if select_major:
            instances = list(map(lambda item: item[0], helper.select_hosts(data, 50).values.tolist()))
        else:
            instances = data['src_ip'].unique().tolist()
        print('Number of hosts to be processed: ' + str(len(instances)))
    else:
        if select_major:
            instances = helper.select_connections(data, 50).values.tolist()
        else:
            instances = data.groupby(['src_ip', 'dst_ip']).size().reset_index().values.tolist()
        print('Number of connections to be processed: ' + str(len(instances)))
    # extract the data per host
    for instance in instances:
        if host_level:
            instance_name = instance
            print('Extracting traces for host ' + instance_name)
            instance_data = data.loc[data['src_ip'] == instance].sort_values(by='date').reset_index(drop=True)
            print('The number of flows for this host are: ' + str(instance_data.shape[0]))
        else:
            instance_name = instance[0] + '-' + instance[1]
            print('Extracting traces for connection ' + instance_name)
            instance_data = data.loc[(data['src_ip'] == instance[0]) & (data['dst_ip'] == instance[1])].\
                sort_values(by='date').reset_index(drop=True)
            print('The number of flows for this connection are: ' + str(instance_data.shape[0]))

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

        # secondly ask if new features has been added during training
        new_features = int(input('Were there any new features added during training (no: 0 | yes: 1)? '))

        # 0 for multivariate | 2 for symbolic
        aggregation = int(input('Do you want to use aggregation windows (no: 0 | yes-rolling: 1 | yes-resample: 2 )? '))
        if aggregation:
            # set the traces output filepath depending on the aggregation value
            # if aggregation has been set to 1 then proper naming is conducted in the extract_traces function of
            # the helper.py file
            resample = False if aggregation == 1 else True
            aggregation = True
            if resample:
                traces_filepath = '/'.join(testing_filepath.split('/')[0:2]) + '/test/' + analysis_type + '/' \
                                  + '_'.join(old_selected) + '/' + testing_filepath.split('/')[2] + '-' + \
                                  instance_name + '-traces_resampled' + ('' if new_features else '_reduced') + \
                                  ('_bdr' if bidirectional else '') + timed_name + '.txt'
            else:
                traces_filepath = '/'.join(testing_filepath.split('/')[0:2]) + '/test/' + analysis_type + '/' \
                                  + '_'.join(old_selected) + '/' + testing_filepath.split('/')[2] + '-' + \
                                  instance_name + '-traces_aggregated' + ('' if new_features else '_reduced') + \
                                  ('_bdr' if bidirectional else '') + timed_name + '.txt'
            # add also the destination ip in case of aggregation
            if host_level:
                selected += ['dst_ip'] if not resample else ['dst_ip', 'date']
            else:
                if resample:
                    selected += ['date']
        # if no new features have been added
        else:
            traces_filepath = '/'.join(testing_filepath.split('/')[0:2]) + '/test/' + analysis_type + '/' + \
                              '_'.join(old_selected) + '/' + testing_filepath.split('/')[2] + '-' + \
                              instance_name + '-traces' + ('_bdr' if bidirectional else '') + timed_name + '.txt'
            aggregation = False
            resample = False

        # create the directory if it does not exist
        os.makedirs(os.path.dirname(traces_filepath), exist_ok=True)

        # and extract the traces
        helper.extract_traces(instance_data, traces_filepath, selected, alphabet_size, timed=timed, dynamic=dynamic,
                              aggregation=aggregation, resample=resample, new_features=bool(new_features))
        # finally reset the selected features
        selected = deepcopy(old_selected)
