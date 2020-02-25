import pandas as pd
from pandas.tseries.offsets import DateOffset
from copy import deepcopy
from scipy.stats import mode


def convert2flexfringe_format(win_data):
    """
    Function to convert the windowed data into a trace in the format accepted by the multivariate version of flexfringe
    :param win_data: the windowed dataframe
    :return: a list of the events in the trace with features separated by comma in each event
    """
    return list(map(lambda x: ','.join(map(lambda t: str(int(t)), x)), win_data.to_numpy().tolist()))


def aggregate_in_windows(data, window):
    """
    Function for aggregating specific features of a dataframe in rolling windows of length window
    Currently the following features are taken into account: source port, destination ip/port, originator's bytes,
    responder's bytes, duration, and protocol
    :param data: the input dataframe
    :param window: the window length
    :return: a dataframe with the aggregated features
    """
    # TODO: maybe roll the window on timestamps instead of indices
    old_column_names = deepcopy(data.columns.values)
    if 'orig_ip_bytes' in old_column_names:
        data['median_orig_bytes'] = data['orig_ip_bytes'].rolling(window).median()
        data['var_orig_bytes'] = data['orig_ip_bytes'].rolling(window).var()
    if 'resp_ip_bytes' in old_column_names:
        data['median_resp_bytes'] = data['resp_ip_bytes'].rolling(window).median()
        data['var_resp_bytes'] = data['resp_ip_bytes'].rolling(window).var()
    if 'duration' in old_column_names:
        data['median_duration'] = data['duration'].rolling(window).median()
        data['var_duration'] = data['duration'].rolling(window).var()
    if 'dst_ip' in old_column_names:
        data['unique_dst_ips'] = data['dst_ip'].rolling(window).apply(pd.nunique)
    if 'src_port' in old_column_names:
        data['unique_src_ports'] = data['src_port'].rolling(window).apply(pd.nunique)
        data['var_src_ports'] = data['src_port'].rolling(window).var()
    if 'dst_port' in old_column_names:
        data['unique_dst_ports'] = data['dst_port'].rolling(window).apply(pd.nunique)
        data['var_dst_ports'] = data['dst_port'].rolling(window).var()
    if 'protocol_num' in old_column_names:
        data['argmax_protocol_num'] = data['protocol_num'].rolling(window).apply(lambda x: mode(x)[0])  # TODO: unsure
        data['var_protocol_num'] = data['protocol_num'].rolling(window).var()
    data.drop(columns=old_column_names, inplace=True)
    return data


def extract_traces(in_filepath, out_filepath, selected, window=5, stride=1, aggregation=0):
    """
    Function for extracting traces from the dataframe stored in the in_filepath and saving them in out_filepath. The
    features to be taken into account are provided in the selected list. Each trace is extracted by rolling a window of
    window seconds in the input data with a stride of stride seconds. If aggregation flag is set to 1, then aggregation
    windows are created in each rolling window
    :param in_filepath: the relative path of the input dataframe
    :param out_filepath: the relative path of the output traces' file
    :param selected: the features to be used
    :param window: the window size
    :param stride: the stride size
    :param aggregation: the aggregation flag - if set to 1, then aggregation windows are created
    :return: creates and stores the traces' file extracted from the input dataframe
    """
    data = pd.read_pickle(in_filepath)

    # set the initial start and end dates, as well as the empty traces' list
    start_date = data['date'].iloc[0]
    end_date = start_date + DateOffset(seconds=window)
    traces = []     # list of lists
    cnt = 0
    tot = len(data.index)   # just for progress visualization purposes

    # iterate through the input dataframe until the end date is greater than the last date recorded
    while end_date < data['date'].iloc[-1]:
        # retrieve the window of interest
        time_mask = (data['date'] >= start_date) & (data['date'] <= end_date)
        windowed_data = data[time_mask]
        if len(windowed_data.index.tolist()) != 0:
            # create aggregated features if needed (currently with a hard-coded window length)
            if aggregation:
                windowed_data = aggregate_in_windows(windowed_data[selected].copy(deep=True),
                                                     min(10, int(len(windowed_data.index))))
                selected = windowed_data.columns.values
            # extract the trace of this window and add it to the traces' list
            traces += [convert2flexfringe_format(windowed_data[selected])]
            # update the progress variable
            cnt = windowed_data.index.tolist()[-1]
        # increment the window limits
        start_date += DateOffset(seconds=stride)
        end_date = start_date + DateOffset(seconds=window)

        # show progress
        # if int((cnt / tot) * 100) // 10 != 0 and int((cnt / tot) * 100) % 10 == 0:
        # print(str(int((cnt / tot) * 100)) + '% of the data processed...')
        print(str(cnt) + '  rows processed...')

    print('Finished with rolling windows!!!')
    print('Starting writing traces to file...')
    # create the traces' file in the needed format
    f = open(out_filepath + "/training_traces.txt", "w")
    f.write(str(len(traces)) + ' ' + '100:' + str(len(selected)) + '\n')
    for trace in traces:
        f.write('1 ' + str(len(trace)) + ' 0:' + ' 0:'.join(trace) + '\n')
    f.close()
    print('Traces written successfully to file!!!')
