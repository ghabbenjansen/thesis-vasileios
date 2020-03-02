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
        # set the needed filepaths
        training_filepath = input('Give the relative path of the dataframe to be used for training: ')
        traces_filepath = '/'.join(training_filepath.split('/')) + '/training/' + training_filepath.split('/')[2] + '-'

        # set the features to be used in the multivariate modelling
        selected = ['src_port', 'dst_port', 'protocol_num', 'orig_ip_bytes', 'resp_ip_bytes']

        # extract the traces and save them in the traces_filepath - the window and the stride sizes of the sliding
        # window can be also specified
        window, stride = helper.set_windowing_vars(training_filepath)
        helper.extract_traces(training_filepath, traces_filepath, selected, window=window, stride=stride)
    else:
        # in case no traces_filepath has been provided (for example if it has already been created) provide one
        traces_filepath = input('Give the path to the input file for flexfringe: ')

    # and set the flags for flexfringe
    extra_args = input('Give any flag arguments for flexfinge in a key value way separated by comma in between e.g. '
                       'key1:value1,ke2:value2,...: ').split(',')

    # run flexfringe to produce the automaton and plot it
    data = flexfringe(traces_filepath, **dict([arg.split(':') for arg in extra_args]))
    show(data)
