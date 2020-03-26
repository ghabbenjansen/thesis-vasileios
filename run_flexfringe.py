import subprocess
import os
import glob
import graphviz
import helper
from connection_clustering import select_hosts
import pandas as pd
import re
from copy import deepcopy

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
    testing = int(input('Training or testing (CTU13: 0 | testing: 1)? '))
    flag = 'CTU-bi'

    if not testing:
        # only if it is for CTU13 this question will be asked
        with_trace = int(input('Is there a trace file (no: 0 | yes: 1)? '))
    else:
        # otherwise we need to create the trace file
        with_trace = 0

    if not with_trace:
        # set the features to be used in the multivariate modelling
        if flag == 'CTU-bi':
            selected = [
                # 'src_port'
                # , 'dst_port'
                'protocol_num'
                , 'src_bytes'
                , 'dst_bytes'
                        ]
        else:
            selected = [
                # 'src_port'
                # , 'dst_port'
                # , 'protocol_num'
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
            # for testing keep only hosts that have at least 2 flows so that enough information is available
            data = select_hosts(data, 2)
            # extract the data per host
            for host in data['src_ip'].unique():
                print('Extracting traces for host ' + host)
                host_data = data[data['src_ip'] == host].sort_values(by='date').reset_index(drop=True)
                print('The number of flows for this host are: ' + str(host_data.shape[0]))

                # extract the traces and save them in the traces' filepath - the window and the stride sizes of the
                # sliding window, as well as the aggregation capability, can be also specified
                window, stride = helper.set_windowing_vars(host_data)
                aggregation = int(input('Do you want to use aggregation windows (no: 0 | yes-rolling: 1 | yes-resample:'
                                        ' 2 )? '))

                # set the traces output filepath depending on the aggregation value
                # if aggregation has been set to 1 then proper naming is conducted in the extract_traces function of the
                # helper.py file
                if not aggregation:
                    traces_filepath = '/'.join(testing_filepath.split('/')[0:2]) + '/test/' + '_'.join(old_selected) + \
                                      '/' + testing_filepath.split('/')[2] + '-' + host + '-traces.txt'
                    aggregation = False
                    resample = False
                else:
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

                # create the directory if it does not exist
                os.makedirs(os.path.dirname(traces_filepath), exist_ok=True)

                # set the trace limits according to the number of flows in the examined dataset
                min_trace_len = int(max(host_data.shape[0] / 10000, 10))
                max_trace_len = int(max(host_data.shape[0] / 100, 1000))
                if host_data.shape[0] < min_trace_len:
                    min_trace_len = host_data.shape[0]
                helper.extract_traces(host_data, traces_filepath, selected, window=window, stride=stride,
                                      trace_limits=(min_trace_len, max_trace_len), dynamic=True,
                                      aggregation=aggregation, resample=resample)
                # finally reset the selected features
                selected = deepcopy(old_selected)

        else:

            # set the input filepath
            training_filepath = input('Give the relative path of the dataframe to be used for CTU13: ')

            if flag == 'CTU-bi':
                data = pd.read_pickle(training_filepath + '/binetflow_normal.pkl')
            else:
                data = pd.read_pickle(training_filepath + '/zeek_normal.pkl')

            # select only hosts with significant number of flows (currently over 200)
            data = select_hosts(data, 200)

            # initialize an empty list to hold the filepaths of the trace files for each host
            traces_filepaths = []

            # extract the data per host
            for host in data['src_ip'].unique():
                print('Extracting traces for host ' + host)
                host_data = data[data['src_ip'] == host].sort_values(by='date').reset_index(drop=True)
                print('The number of flows for this host are: ' + str(host_data.shape[0]))

                # extract the traces and save them in the traces' filepath - the window and the stride sizes of the
                # sliding window, as well as the aggregation capability, can be also specified
                window, stride = helper.set_windowing_vars(host_data)
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
                                          host + '-traces_resampled.txt'
                    else:
                        traces_filepath = '/'.join(training_filepath.split('/')[0:2]) + '/training/' + \
                                          '_'.join(old_selected) + '/' + training_filepath.split('/')[2] + '-' + \
                                          host + '-traces_aggregated.txt'
                    # add also the destination ip in case of aggregation
                    selected += ['dst_ip'] if not resample else ['dst_ip', 'date']

                # create the directory if it does not exist
                os.makedirs(os.path.dirname(traces_filepath), exist_ok=True)

                # set the trace limits according to the number of flows in the examined dataset
                # min_trace_len = int(max(host_data.shape[0] / 10000, 10))
                # max_trace_len = int(max(host_data.shape[0] / 100, 1000))
                helper.extract_traces(host_data, traces_filepath, selected, window=window, stride=stride,
                                      trace_limits=(10, 500), dynamic=True,
                                      aggregation=aggregation, resample=resample)

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
