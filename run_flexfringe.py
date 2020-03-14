import subprocess
import os
import glob
import graphviz
import helper
from connection_clustering import select_hosts
import pandas as pd

# TODO: change the filepath from hardcoded to input setting
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
    print("%s" % subprocess.run([filepath+"flexfringe", ] + command + [args[0]], stdout=subprocess.PIPE, check=True)
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
    # TODO: make it less case sensitive in case of other dataset
    extension = '-'.join(args[0].split('/')[-1].split('-')[0:4])
    new_file_name = "final_" + extension + "_dfa.dot"
    new_file = os.path.join("outputs", new_file_name)
    os.rename(old_file, new_file)

    # and open the output dot file
    try:
        with open("outputs/" + new_file_name) as fh:
            return fh.read(), 'outputs/' + new_file_name
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
        directory = filepath.split('/')[0]
        filename = '_'.join(filepath.split('/')[1].split('_')[1:])  # first remove the front 'final_'
        filename = '.'.join(filename.split('.')[:-1])   # and then remove the '.dot' ending
        # and then create the dfa plot
        g = graphviz.Source(data, filename=filename, directory=directory, format="png")
        g.render(filename=filename, directory=directory, view=True, cleanup=True)


if __name__ == '__main__':
    # first check why we want to run flexfringe (training or testing - in the second case only the traces are extracted)
    # check if there is a need to create the trace file or there is already there
    testing = int(input('Training or testing?? (training: 0 | testing: 1)? '))

    if not testing:
        # only if it is for training this question will be asked
        with_trace = int(input('Is there a trace file (no: 0 | yes: 1)? '))
    else:
        # otherwise we need to create the trace file
        with_trace = 0

    # TODO: update the trace extraction part for the testing datasets too (This entails keeping the number of flows)
    if not with_trace:
        # set the features to be used in the multivariate modelling
        selected = ['src_port', 'dst_port', 'protocol_num', 'orig_ip_bytes', 'resp_ip_bytes']

        # set a mapping between features used and numbers for better identification of the traces' content
        feature_mapping = {
            'src_port': 0,
            'dst_port': 1,
            'protocol_num': 2,
            'orig_ip_bytes': 3,
            'resp_ip_bytes': 4,
            'duration': 5
        }

        if testing:
            # set the input filepath
            testing_filepath = input('Give the relative path of the dataframe to be used for tesing: ')
            data = pd.read_pickle(testing_filepath)
            # extract the data per host
            for host in data['src_ip'].unique():
                print('Extracting traces for host ' + host)
                host_data = data[data['src_ip'] == host]
                host_data.reset_index(drop=True, inplace=True)
                print('The number of flows for this host are: ' + str(host_data.shape[0]))

                # extract the traces and save them in the traces' filepath - the window and the stride sizes of the
                # sliding window, as well as the aggregation capability, can be also specified
                window, stride = helper.set_windowing_vars(host_data)
                aggregation = int(input('Do you want to use aggregation windows (no: 0 | yes: 1)? '))

                # set the traces output filepath depending on the aggregation value
                # if aggregation has been set to 1 then proper naming is conducted in the extract_traces function of the
                # helper.py file
                if not aggregation:
                    traces_filepath = '/'.join(testing_filepath.split('/')[0:2]) + '/test/' + \
                                      testing_filepath.split('/')[2] + '-' + host + '-traces-' + \
                                      '-'.join([str(feature_mapping[feature]) for feature in selected]) + '.txt'
                else:
                    traces_filepath = '/'.join(testing_filepath.split('/')[0:2]) + '/test/' + \
                                      testing_filepath.split('/')[2] + '-' + host + '-traces.txt'

                helper.extract_traces(host_data, traces_filepath, selected, window=window, stride=stride,
                                      trace_limits=(100, 6000), dynamic=True, aggregation=aggregation)

        else:

            # set the input filepath
            training_filepath = input('Give the relative path of the dataframe to be used for training: ')

            # select only hosts with significant number of flows (currently over 50)
            data = select_hosts(pd.read_pickle(training_filepath), 200)

            # initialize an empty list to hold the filepaths of the trace files for each host
            traces_filepaths = []

            # extract the data per host
            for host in data['src_ip'].unique():
                print('Extracting traces for host ' + host)
                host_data = data[data['src_ip'] == host]
                host_data.reset_index(drop=True, inplace=True)
                print('The number of flows for this host are: ' + str(host_data.shape[0]))

                # extract the traces and save them in the traces' filepath - the window and the stride sizes of the
                # sliding window, as well as the aggregation capability, can be also specified
                window, stride = helper.set_windowing_vars(host_data)
                aggregation = int(input('Do you want to use aggregation windows (no: 0 | yes: 1)? '))

                # set the traces output filepath depending on the aggregation value
                # if aggregation has been set to 1 then proper naming is conducted in the extract_traces function of the
                # helper.py file
                if not aggregation:
                    traces_filepath = '/'.join(training_filepath.split('/')[0:2]) + '/training/' + \
                                      training_filepath.split('/')[2] + '-' + host + '-traces-' + \
                                      '-'.join([str(feature_mapping[feature]) for feature in selected]) + '.txt'
                else:
                    traces_filepath = '/'.join(training_filepath.split('/')[0:2]) + '/training/' + \
                                      training_filepath.split('/')[2] + '-' + host + '-traces.txt'

                helper.extract_traces(host_data, traces_filepath, selected, window=window, stride=stride,
                                      trace_limits=(100, 6000), dynamic=True, aggregation=aggregation)

                # add the trace filepath of each host's traces to the list
                traces_filepaths += [traces_filepath]
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
