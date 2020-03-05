import subprocess
import graphviz
import helper

# TODO: change the filepath from hardcoded to input setting
filepath = '/Users/vserentellos/Documents/dfasat/'


def flexfringe(*args, **kwargs):
    """
    Wrapper function for running flexfringe in Python
    :param args: the input arguments for flexfringe
    :param kwargs: the keywords arguments (flags with their values) for flexfringe
    :return: the opened dot file of the output of flexfringe
    """
    command = ["--help"]

    # collect the keyword arguments
    if len(kwargs) > 1:
        command = ["-" + key + "=" + kwargs[key] for key in kwargs]

    # run flexfringe with the given arguments
    print("%s" % subprocess.run([filepath+"flexfringe", ] + command + [args[0]], stdout=subprocess.PIPE, check=True)
          .stdout.decode())

    # and open the output dot file
    try:
        with open("outputs/final.json") as fh:
            return fh.read()
    except FileNotFoundError:
        pass

    return "No output file was generated."


def show(data):
    """
    Function for plotting a dot file, created after a run of flexfringe, through graphviz
    :param data: the content of the dot file
    :return: plots the created automaton provided in data
    """
    if data == "":
        pass
    else:
        g = graphviz.Source(data, format="png")
        g.render(view=True)


if __name__ == '__main__':
    # check if there is a need to create the trace file or there is already there
    with_trace = int(input('Is there a trace file (no: 0 | yes: 1)? '))

    if not with_trace:
        # set the input filepath
        training_filepath = input('Give the relative path of the dataframe to be used for training: ')

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

        # extract the traces and save them in the traces' filepath - the window and the stride sizes of the sliding
        # window, as well as the aggregation capability, can be also specified
        window, stride = helper.set_windowing_vars(training_filepath, 500)
        aggregation = int(input('Do you want to use aggregation windows (no: 0 | yes: 1)? '))

        # set the traces output filepath depending on the aggregation value
        # if aggregation has been set to 1 then proper naming is conducted in the extract_traces function of the
        # helper.py file
        if not aggregation:
            traces_filepath = '/'.join(training_filepath.split('/')[0:2]) + '/training/' + \
                              training_filepath.split('/')[2] + '-traces-' + '-'.join([str(feature_mapping[feature])
                                                                                      for feature in selected]) + '.txt'
        else:
            traces_filepath = '/'.join(training_filepath.split('/')[0:2]) + '/training/' + \
                              training_filepath.split('/')[2] + '-traces.txt'

        helper.extract_traces(training_filepath, traces_filepath, selected, window=window, stride=stride, dynamic=False,
                              aggregation=aggregation)
    else:
        # in case the traces' filepath already exists, provide it
        traces_filepath = input('Give the path to the input file for flexfringe: ')

    # and set the flags for flexfringe
    extra_args = input('Give any flag arguments for flexfinge in a key value way separated by comma in between e.g. '
                       'key1:value1,ke2:value2,...: ').split(',')

    # run flexfringe to produce the automaton and plot it
    data = flexfringe(traces_filepath, **dict([arg.split(':') for arg in extra_args]))
    show(data)
