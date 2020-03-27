import subprocess
import os
import glob
import graphviz
import helper
from connection_clustering import select_hosts
import pandas as pd
import re
from copy import deepcopy
import pickle

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
    dataset_name = args[0].split('/')[-4]
    extension = re.search('(.+?)-traces', args[0].split('/')[-1]).group(1)
    # add this naming in case aggregation windows have been used
    if 'aggregated' in args[0]:
        extension += '_aggregated'
    if 'resampled' in args[0]:
        extension += '_resampled'
    new_file_name = extension + "_dfa.dot"
    new_file = os.path.join("outputs/" + dataset_name + '/' + features, new_file_name)

    # create the directory if it does not exist and rename the created dot file
    os.makedirs(os.path.dirname(new_file), exist_ok=True)
    os.rename(old_file, new_file)

    # and open the output dot file
    try:
        with open("outputs/" + dataset_name + '/' + features + "/" + new_file_name) as fh:
            return fh.read(), "outputs/" + dataset_name + '/' + features + "/" + new_file_name
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
    # first check why we want to run flexfringe (CTU13 or testing - in the second case only the traces are extracted)
    # check if there is a need to create the trace file or there is already there
    testing = int(input('Training or testing (training: 0 | testing: 1)? '))
    flag = 'CTU-bi'

    if not testing:
        # only if it is for CTU13 this question will be asked
        with_trace = int(input('Is there a trace file (no: 0 | yes: 1)? '))
    else:
        # otherwise we need to create the trace file
        with_trace = 0

    if not with_trace:
        # first check if discretization is enabled
        with_discretization = int(
            input('Discretize numeric features (ports, bytes, duration, packets) (no: 0 | yes: 1)? '))
        # set the features to be used in the multivariate modelling
        if flag == 'CTU-bi':
            selected = [
                # 'src_port'
                # , 'dst_port'
                'protocol_num'
                # , 'duration'
                , 'src_bytes'
                , 'dst_bytes'
                        ]
        else:
            selected = [
                # 'src_port'
                # , 'dst_port'
                # , 'protocol_num'
                # , 'duration'
                'orig_ip_bytes'
                , 'resp_ip_bytes'
            ]
        old_selected = deepcopy(selected)

        if testing:
            # set the input filepath of the dataframes' directory
            testing_filepath = input('Give the relative path of the dataset to be used for testing: ')
            if flag == 'CTU-bi':
                normal = pd.read_pickle(testing_filepath + '/binetflow_normal.pkl')
                anomalous = pd.read_pickle(testing_filepath + '/binetflow_anomalous.pkl')
            else:
                normal = pd.read_pickle(testing_filepath + '/zeek_normal.pkl')
                anomalous = pd.read_pickle(testing_filepath + '/zeek_anomalous.pkl')
            data = pd.concat([normal, anomalous], ignore_index=True).reset_index(drop=True)

            if with_discretization:
                # first retireve the discretization limits to be used for each feature
                discretization_filepath = input('Provide the filepath of the discretization limits: ')
                with open(discretization_filepath, 'rb') as f:
                    discretization_dict = pickle.load(f)
                # then apply discretization to the appropriate features
                for feature in discretization_dict.keys():
                    data[feature + '_num'] = data[feature].apply(helper.find_percentile,
                                                                 args=(discretization_dict[feature],))
                    selected.remove(feature)
                    selected += [feature + '_num']
                old_selected = deepcopy(selected)

            # for testing keep only hosts that have at least 2 flows so that enough information is available
            data = select_hosts(data, 2)
            # extract the data per host
            for host in data['src_ip'].unique():
                print('Extracting traces for host ' + host)
                host_data = data[data['src_ip'] == host].sort_values(by='date').reset_index(drop=True)
                print('The number of flows for this host are: ' + str(host_data.shape[0]))

                # first ask if new features has been added during training
                new_features = int(input('Were there any new features added during training (no: 0 | yes: 1)? '))

                if new_features:
                    # extract the traces and save them in the traces' filepath
                    aggregation = int(input('Do you want to use aggregation windows (no: 0 | yes-rolling: 1 | '
                                            'yes-resample: 2 )? '))

                    # set the traces output filepath depending on the aggregation value
                    # if aggregation has been set to 1 then proper naming is conducted in the extract_traces function of
                    # the helper.py file
                    resample = False if aggregation == 1 else True
                    aggregation = True
                    if resample:
                        traces_filepath = '/'.join(testing_filepath.split('/')[0:2]) + '/test/' + \
                                          '_'.join(old_selected) + '/' + testing_filepath.split('/')[2] + '-' + host + \
                                          '-traces_resampled.txt'
                    else:
                        traces_filepath = '/'.join(testing_filepath.split('/')[0:2]) + '/test/' \
                                          + '_'.join(old_selected) + '/' + testing_filepath.split('/')[2] + '-' + \
                                          host + '-traces_aggregated.txt'
                    # add also the destination ip in case of aggregation
                    selected += ['dst_ip'] if not resample else ['dst_ip', 'date']
                # if no new features have been added
                else:
                    traces_filepath = '/'.join(testing_filepath.split('/')[0:2]) + '/test/' + '_'.join(old_selected) + \
                                      '/' + testing_filepath.split('/')[2] + '-' + host + '-traces.txt'
                    aggregation = False
                    resample = False

                # create the directory if it does not exist
                os.makedirs(os.path.dirname(traces_filepath), exist_ok=True)

                # and extract the traces
                helper.extract_traces(host_data, traces_filepath, selected, dynamic=True, aggregation=aggregation,
                                      resample=resample)
                # finally reset the selected features
                selected = deepcopy(old_selected)

        else:

            # set the input filepath
            training_filepath = input('Give the relative path of the dataframe to be used for training: ')

            if flag == 'CTU-bi':
                data = pd.read_pickle(training_filepath + '/binetflow_normal.pkl')
            else:
                data = pd.read_pickle(training_filepath + '/zeek_normal.pkl')

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

            # select only hosts with significant number of flows (currently over 200)
            data = select_hosts(data, 200)

            # initialize an empty list to hold the filepaths of the trace files for each host
            traces_filepaths = []

            # extract the data per host
            for host in data['src_ip'].unique():
                print('Extracting traces for host ' + host)
                host_data = data[data['src_ip'] == host].sort_values(by='date').reset_index(drop=True)
                print('The number of flows for this host are: ' + str(host_data.shape[0]))

                # first ask if new features are to be added
                new_features = int(input('Are there any new features to be added (no: 0 | yes: 1)? '))

                # extract the traces and save them in the traces' filepath
                aggregation = int(input('Do you want to use aggregation windows (no: 0 | yes-rolling: 1 | yes-resample:'
                                        ' 2 )? '))

                # set the traces output filepath depending on the aggregation value
                # if aggregation has been set to 1 then proper naming is conducted in the extract_traces function of the
                # helper.py file
                if not aggregation:
                    traces_filepath = '/'.join(training_filepath.split('/')[0:2]) + '/training/' + \
                                      '_'.join(old_selected) + '/' + training_filepath.split('/')[2] + '-' + host + \
                                      '-traces.txt'
                    aggregation = False
                    resample = False
                else:
                    resample = False if aggregation == 1 else True
                    aggregation = True
                    if resample:
                        traces_filepath = '/'.join(training_filepath.split('/')[0:2]) + '/training/' + \
                                          '_'.join(old_selected) + '/' + training_filepath.split('/')[2] + '-' + \
                                          host + '-traces_resampled' + ('' if new_features else '_reduced') + '.txt'
                    else:
                        traces_filepath = '/'.join(training_filepath.split('/')[0:2]) + '/training/' + \
                                          '_'.join(old_selected) + '/' + training_filepath.split('/')[2] + '-' + \
                                          host + '-traces_aggregated' + ('' if new_features else '_reduced') + '.txt'
                    # add also the destination ip in case of aggregation
                    selected += ['dst_ip'] if not resample else ['dst_ip', 'date']

                # create the directory if it does not exist
                os.makedirs(os.path.dirname(traces_filepath), exist_ok=True)

                # and extract the traces
                helper.extract_traces(host_data, traces_filepath, selected, dynamic=True, aggregation=aggregation,
                                      resample=resample, new_features=bool(new_features))

                # add the trace filepath of each host's traces to the list
                traces_filepaths += [traces_filepath]
                # and reset the selected features
                selected = deepcopy(old_selected)
    else:
        # in case the traces' filepath already exists, provide it (in this case only one path - NOT a list
        traces_filepaths = [input('Give the path to the input file for flexfringe: ')]

    if not testing:
        # create a model for each host
        for traces_filepath in traces_filepaths:
            # and set the flags for flexfringe
            extra_args = input('Give any flag arguments for flexfinge in a key value way separated by comma in between '
                               'e.g. key1:value1,ke2:value2,...: ').split(',')

            # run flexfringe to produce the automaton and plot it
            modelled_data, storing_path = flexfringe(traces_filepath, **dict([arg.split(':') for arg in extra_args]))
            show(modelled_data, storing_path)
