import pandas as pd
from pandas.tseries.offsets import DateOffset


def convert2flexfringe_format(win_data):
    """
    Function to convert the windowed data into a trace in the format accepted by the multivariate version of flexfringe
    :param win_data: the windowed dataframe
    :return: a list of the events in the trace with features separated by comma in each event
    """
    return list(map(lambda x: ','.join(x), win_data.to_numpy().tolist()))


def extract_traces(in_filepath, out_filepath, selected, window=5, stride=1):
    """
    Function for extracting traces from the dataframe stored in the in_filepath and saving them in out_filepath. The
    features to be taken into account are provided in the selected list. Each trace is extracted by rolling a window of
    window seconds in the input data with a stride of stride seconds
    :param in_filepath: the relative path of the input dataframe
    :param out_filepath: the relative path of the output traces' file
    :param selected: the features to be used
    :param window: the window size
    :param stride: the stride size
    :return: creates and stores the traces' file extracted from the input dataframe
    """
    data = pd.read_pickle(in_filepath)

    # set the initial start and end dates, as well as the empty traces' list
    start_date = data['date'].iloc[0]
    end_date = start_date + DateOffset(seconds=window)
    traces = []     # list of lists

    # iterate through the input dataframe until the end date is greater than the last date recorded
    while end_date < data['date'].iloc[-1]:
        # retrieve the window of interest
        time_mask = (data['date'] >= start_date) & (data['date'] <= end_date)
        windowed_data = data[time_mask]     # TODO: not sure about its validity - To check it
        # extract the trace of this window and add it to the traces' list
        traces += [convert2flexfringe_format(windowed_data[selected])]
        # increment the window limits
        start_date += DateOffset(seconds=stride)
        end_date = start_date + DateOffset(seconds=window)

    # create the traces' file in the needed format
    f = open(out_filepath + "training_traces.txt", "w")
    f.write(str(len(traces)) + ' ' + '100:' + str(len(selected)))   # TODO: check what this 100 number is
    for trace in traces:
        f.write('1 ' + str(len(trace)) + ' 0:' + ' 0:'.join(trace))
    f.close()

