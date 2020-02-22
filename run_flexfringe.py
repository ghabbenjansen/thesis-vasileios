import subprocess
import sys
import graphviz

# TODO: change the filepath from hardcoded to input setting
filepath = '/Users/vserentellos/Documents/dfasat/'


def flexfringe(*args, **kwargs):
    command = ["--help"]

    if len(kwargs) > 1:
        command = ["-" + key + "=" + kwargs[key] for key in kwargs]

    print("%s" % subprocess.run([filepath+"flexfringe", ] + command + [args[0]], stdout=subprocess.PIPE).stdout.decode())

    try:
        with open(filepath + "outputs/final.dot") as fh:
            return fh.read()
    except FileNotFoundError:
        pass

    return "No output file was generated."


def show(data):
    if data == "":
        pass
    else:
        g = graphviz.Source(data, format="png")
        g.render(view=True)


if __name__ == '__main__':
    input_data_path = input('Give the path of the input file for flexfringe: ')
    extra_args = input('Give any flag arguments for flexfinge in a key value way separated by comma in between (e.g. '
                       'key1:value1,ke2:value2,...: ').split(',')
    data = flexfringe(input_data_path, **dict([arg.split(':') for arg in extra_args]))
    show(data)
