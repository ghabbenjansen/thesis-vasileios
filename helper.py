from pandas.tseries.offsets import DateOffset
from copy import deepcopy
from scipy.stats import mode
from statistics import mean
from model import ModelNode, Model
from tslearn.metrics import dtw
from sklearn.preprocessing import MinMaxScaler
import re
import pickle
import pandas as pd


def set_windowing_vars(data):
    """
    Function for automatically calculating an initial estimation of the time windows and strides to be used for creating
    the traces, as the median of the time differences between the flows in the dataframe
    :param data: the input dataframe
    :return: a tuple with the calculated time windows and strides in a dataframe format
    """
    # find the median of the time differences in the dataframe
    median_diff = data['date'].sort_values().diff().median()
    return 25 * median_diff, 5 * median_diff


def traces_dissimilarity(trace1, trace2, multivariate=True, normalization=True):
    """
    Function for calculating the dissimilarity between two input traces. The traces are in the form of list of lists and
    are dealt either as multivariate series or as multiple univariate series depending on the value of the multivariate
    flag provided
    :param trace1: the first trace
    :param trace2: the second trace
    :param multivariate: the multivariate flag
    :param normalization: the normalization flag for performing (or not) min-max normalization
    :return: the dissimilarity score (the lower the score the higher the similarity)
    """
    if normalization:
        traces = MinMaxScaler().fit_transform(trace1 + trace2)
        trace1 = traces[:len(trace1)].tolist()
        trace2 = traces[len(trace1):].tolist()
    return dtw(trace1, trace2) if multivariate else mean([dtw(list(list(zip(*trace1))[j]), list(list(zip(*trace2))[j]))
                                                          for j in range(len(trace1[0]))])


def convert2flexfringe_format(win_data, ints=True):
    """
    Function to convert the windowed data into a trace in the format accepted by the multivariate version of flexfringe
    :param win_data: the windowed dataframe
    :param ints: flag showing if there are only int data in the dataframe
    :return: a list of the events in the trace with features separated by comma in each event
    """
    fun = lambda x: int(x) if ints else float(x)
    return list(map(lambda x: ','.join(map(lambda t: str(fun(t)), x)), win_data.to_numpy().tolist()))


def trace2list(trace):
    """
    Function for converting a list of string records of a trace to a list of lists
    :param trace: the list with the string records
    :return: the converted list of lists
    """
    return list(map(lambda x: list(map(int, x.split(','))), trace))


def calculate_window_mask(data, start_date, end_date):
    """
    Function for calculating the window mask for the input dataframe given a starting and an ending date
    :param data: the input dataframe
    :param start_date: the starting date
    :param end_date: the ending date
    :return: the window mask
    """
    return (data['date'] >= start_date) & (data['date'] <= end_date)


def aggregate_in_windows(data, window, timed=False, resample=False):
    """
    Function for aggregating specific features of a dataframe in rolling windows of length window
    Currently the following features are taken into account: source port, destination ip/port, originator's bytes,
    responder's bytes, duration, and protocol
    :param data: the input dataframe
    :param window: the window length
    :param timed: boolean flag specifying if aggregation window should take into account the timestamps
    :param resample: boolean flag specifying if aggregation window should be rolling or resampling
    :return: a dataframe with the aggregated features
    """
    old_column_names = deepcopy(data.columns.values)
    # if the timed flag is True then timestamps are used as indices
    if timed:
        data.set_index('date', inplace=True)
    if not resample:
        if 'orig_ip_bytes' in old_column_names:
            data['median_orig_bytes'] = data['orig_ip_bytes'].rolling(window, min_periods=1).median()
            data['std_orig_bytes'] = data['orig_ip_bytes'].rolling(window, min_periods=1).std()
        if 'resp_ip_bytes' in old_column_names:
            data['median_resp_bytes'] = data['resp_ip_bytes'].rolling(window, min_periods=1).median()
            data['std_resp_bytes'] = data['resp_ip_bytes'].rolling(window, min_periods=1).std()
        if 'duration' in old_column_names:
            data['median_duration'] = data['duration'].rolling(window, min_periods=1).median()
            data['std_duration'] = data['duration'].rolling(window, min_periods=1).std()
        if 'dst_ip' in old_column_names:
            data['unique_dst_ips'] = pd.DataFrame(pd.Categorical(data['dst_ip']).codes, index=data.index).\
                rolling(window, min_periods=1).apply(lambda x: len(set(x)), raw=False)
        if 'src_port' in old_column_names:
            data['unique_src_ports'] = data['src_port'].rolling(window, min_periods=1).apply(lambda x: len(set(x)),
                                                                                             raw=False)
            data['std_src_ports'] = data['src_port'].rolling(window, min_periods=1).std()
        if 'dst_port' in old_column_names:
            data['unique_dst_ports'] = data['dst_port'].rolling(window, min_periods=1).\
                apply(lambda x: len(set(x)), raw=False)
            data['std_dst_ports'] = data['dst_port'].rolling(window, min_periods=1).std()
        if 'protocol_num' in old_column_names:
            data['argmax_protocol_num'] = data['protocol_num'].rolling(window, min_periods=1).\
                apply(lambda x: mode(x)[0], raw=False)
            data['std_protocol_num'] = data['protocol_num'].rolling(window, min_periods=1).std()
        data.drop(columns=old_column_names, inplace=True)
        data.bfill(axis='rows', inplace=True)
    else:
        # can be called only if timed flag has been set to True
        frames = []
        new_column_names = []
        if 'orig_ip_bytes' in old_column_names:
            new_column_names += ['median_orig_bytes', 'std_orig_bytes']
            frames += [data['orig_ip_bytes'].resample(window).median(), data['orig_ip_bytes'].resample(window).std()]
        if 'resp_ip_bytes' in old_column_names:
            new_column_names += ['median_resp_bytes', 'std_resp_bytes']
            frames += [data['resp_ip_bytes'].resample(window).median(), data['resp_ip_bytes'].resample(window).std()]
        if 'duration' in old_column_names:
            new_column_names += ['median_duration', 'std_duration']
            frames += [data['duration'].resample(window).median(), data['duration'].resample(window).std()]
        if 'dst_ip' in old_column_names:
            new_column_names += ['unique_dst_ips']
            frames += [data['dst_ip'].resample(window).nunique()]
        if 'src_port' in old_column_names:
            new_column_names += ['unique_src_ports', 'std_src_ports']
            frames += [data['src_port'].resample(window).nunique(), data['src_port'].resample(window).std()]
        if 'dst_port' in old_column_names:
            new_column_names += ['unique_dst_ports', 'std_dst_ports']
            frames += [data['dst_port'].resample(window).nunique(), data['dst_port'].resample(window).std()]
        if 'protocol_num' in old_column_names:
            new_column_names += ['argmax_protocol_num', 'std_protocol_num']
            frames += [data['protocol_num'].resample(window).apply(lambda x: mode(x)[0]),
                       data['protocol_num'].resample(window).std()]
        data = pd.concat(frames, axis=1)
        data.columns = new_column_names
        data.dropna(inplace=True)
    return data


def extract_traces(data, out_filepath, selected, window, stride, trace_limits, dynamic=True, aggregation=False,
                   resample=False):
    """
    Function for extracting traces from the imput dataframe and saving them in out_filepath. The features to be taken
    into account are provided in the selected list. Each trace is extracted by rolling a window of window seconds
    in the input data with a stride of stride seconds. If dynamic flag is set to True, then a dynamically changing
    window is used instead. If aggregation flag is set to True, then aggregation windows are created in each rolling
    window.
    :param data: the input dataframe
    :param out_filepath: the relative path of the output traces' file
    :param selected: the features to be used
    :param window: the window size
    :param stride: the stride size
    :param trace_limits: a tuple containing the minimum and maximum length that a trace can have
    :param dynamic: boolean flag about the use of dynamically changing windows
    :param aggregation: the aggregation flag - if set to True, then aggregation windows are created
    :param resample: the resampling flag - if set to True, then resampling is used in the aggregation windows
    :return: creates and stores the traces' file extracted from the input dataframe
    """

    # create an anonymous function for increasing timestamps given the type of the window (int or Timedelta)
    time_inc = lambda x, w: x + DateOffset(seconds=w) if type(window) == int else x + w

    # set the initial start and end dates, as well as the empty traces' list and the window limits
    start_date = data['date'].iloc[0]
    end_date = time_inc(start_date, window)
    traces = []  # list of lists
    traces_indices = []  # list of lists for storing the indices of the flows contained in each trace
    # the minimum and maximum indices of the time window under consideration
    # two values are used for the indices of two consecutive windows
    min_idx = [-2, -1]
    max_idx = [-2, -1]
    # structures just for progress visualization purposes
    cnt = 0
    tot = len(data.index)
    progress_list = []
    # extract the traces' limits
    min_trace_length, max_trace_length = trace_limits
    # create a dict for testing if all the flows have been included in the traces
    assertion_dict = dict(zip(data.index.tolist(), len(data.index.tolist()) * [False]))
    # keep a copy of the actually selected features in case aggregation is used
    old_selected = deepcopy(selected)
    # keep also a variable of the number of features to be used for the model creation
    num_of_features = len(selected)

    # iterate through the input dataframe until the end date is greater than the last date recorded
    while end_date < data['date'].iloc[-1]:
        # retrieve the window of interest
        time_mask = calculate_window_mask(data, start_date, end_date)
        windowed_data = data[time_mask]
        window_len = len(windowed_data.index.tolist())
        # if there is at least one record in the window
        if window_len != 0:
            if dynamic:
                # store the minimum and maximum indices of the time window to evaluate how much it moved
                if min_idx[0] == -1:  # the case of the first recorded time window
                    min_idx[0] = data.index[time_mask].tolist()[0]
                    max_idx[0] = data.index[time_mask].tolist()[-1]
                elif min_idx[1] == -1:  # the case of the second recorded time window
                    min_idx[1] = data.index[time_mask].tolist()[0]
                    max_idx[1] = data.index[time_mask].tolist()[-1]
                else:  # otherwise update the previous values and add the new ones
                    min_idx[0] = deepcopy(min_idx[1])
                    max_idx[0] = deepcopy(max_idx[1])
                    min_idx[1] = data.index[time_mask].tolist()[0]
                    max_idx[1] = data.index[time_mask].tolist()[-1]

                # first check if the time window captured new information
                while min_idx[0] == min_idx[1] and max_idx[0] == max_idx[1]:
                    print('-------------- No change between traces ==> Increasing the stride... --------------')
                    start_date = time_inc(start_date, stride)
                    end_date = time_inc(start_date, window)
                    time_mask = calculate_window_mask(data, start_date, end_date)
                    window_len = len(data[time_mask].index.tolist())
                    # if the new window is empty or we have surpassed the next unseen flow, the next window is set to
                    # start at the timestamp of this unseen flow
                    if window_len == 0 or data.index[time_mask].tolist()[0] > max_idx[1] + 1:
                        start_date = data['date'].iloc[max_idx[1] + 1]
                        end_date = time_inc(start_date, window)
                        time_mask = calculate_window_mask(data, start_date, end_date)
                        window_len = len(data[time_mask].index.tolist())
                    # set the updated indices
                    min_idx[1] = data.index[time_mask].tolist()[0]
                    max_idx[1] = data.index[time_mask].tolist()[-1]
                    # and increase the stride in case we still haven't captured new information
                    stride *= 2
                    if stride >= window:
                        window = stride * 5

                # set the parameters for the length adjustment process of each trace
                magnifier = 2
                reducer = 0.05
                # check that the trace length conforms to the specified limits
                while window_len < min_trace_length or window_len > max_trace_length:
                    # first check the case of a very large window
                    while window_len > max_trace_length:
                        print('-------------- Too many flows in the trace ==> Reducing time window... --------------')
                        window /= magnifier
                        if stride >= window:
                            stride = window / 5
                        end_date = time_inc(start_date, window)
                        time_mask = calculate_window_mask(data, start_date, end_date)
                        window_len = len(data[time_mask].index.tolist())

                    # then check the case of a very small window
                    while window_len < min_trace_length:
                        print('-------------- Too few flows in the trace ==> Increasing time window... --------------')
                        window *= magnifier
                        end_date = time_inc(start_date, window)
                        time_mask = calculate_window_mask(data, start_date, end_date)
                        window_len = len(data[time_mask].index.tolist())
                        # limit case to prevent integer overflow in the window size
                        if end_date > data['date'].iloc[-1]:
                            break

                    # and update the window indices
                    if min_idx[0] < 0:
                        min_idx[0] = data.index[time_mask].tolist()[0]
                        max_idx[0] = data.index[time_mask].tolist()[-1]
                    else:
                        min_idx[1] = data.index[time_mask].tolist()[0]
                        max_idx[1] = data.index[time_mask].tolist()[-1]

                    # update the magnifier in case more iterations are needed due to fluctuations
                    magnifier -= reducer
                    # in case that the fluctuations cannot be dealt with the current values, refine them and start over
                    if magnifier == 1:
                        magnifier = 2
                        reducer = reducer/2

                    # limit case to prevent endless loop
                    if end_date > data['date'].iloc[-1]:
                        break

                # finally get the current window
                windowed_data = data[time_mask]

            # insert the indices of the current trace to the assertion dictionary
            assertion_dict.update(zip(data.index[time_mask].tolist(), len(data.index[time_mask].tolist()) * [True]))

            # create aggregated features if needed (currently with a hard-coded window length)
            if aggregation:
                aggregation_length = '5S' if resample else min(10, int(len(windowed_data.index)))
                timed = True if resample else False
                windowed_data = aggregate_in_windows(windowed_data[selected].copy(deep=True), aggregation_length, timed,
                                                     resample)
                selected = windowed_data.columns.values
                num_of_features = len(selected)

            # extract the trace of this window and add it to the traces' list
            ints = False if aggregation or 'duration' in old_selected else True
            # this case applies only on resampling in case there are no more than 1 flow per resampling window
            # TODO: maybe check if flows are missed when resampling is used
            if windowed_data.shape[0] != 0:
                traces += [convert2flexfringe_format(windowed_data[selected], ints)]
            selected = deepcopy(old_selected)
            # store also the flow indices of the current time window
            if windowed_data.shape[0] != 0:     # this case applies only on resampling as explained above
                traces_indices += [windowed_data.index.tolist()]

            # old implementation of window dissimilarity (not used now)
            # dissim = traces_dissimilarity(deepcopy(trace2list(traces[-1])), deepcopy(trace2list(traces[-2])))

            # update the progress variable
            cnt = data.index[time_mask].tolist()[-1]

            # increment the window limits
            start_date = time_inc(start_date, stride)
            end_date = time_inc(start_date, window)
        # if there are no records in the window
        else:
            if dynamic:
                # if there is no observation in the window just set as start date the one of the next non-visited index
                print('------------- No records in the trace ==> Proceeding to the next recorded flow... -------------')
                # check if there are less than two windows captured
                if max_idx[1] < 0:
                    # mostly to catch an implementation error since at least the first window should have records
                    if max_idx[0] < 0:
                        print('This should not happen!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        start_date = data['date'].iloc[0]
                    else:
                        start_date = data['date'].iloc[max_idx[0]+1]
                # otherwise set the start date of the last visited index + 1
                else:
                    start_date = data['date'].iloc[max_idx[1] + 1]
                end_date = time_inc(start_date, window)
            else:
                # increment the window limits
                start_date = time_inc(start_date, stride)
                end_date = time_inc(start_date, window)

        # show progress
        prog = int((cnt / tot) * 100)
        if prog // 10 != 0 and prog // 10 not in progress_list:
            progress_list += [prog // 10]
            print('More than ' + str((prog // 10) * 10) + '% of the data processed...')

    # addition of the last flows in the dataframe in case they weren't added
    if not all(list(assertion_dict.values())):
        time_mask = calculate_window_mask(data, start_date, end_date)
        windowed_data = data[time_mask]
        # in case that the start date is also greater than the last seen flow then set the start date appropriately
        if windowed_data.index.tolist()[0] > max_idx[1] + 1:
            start_date = data['date'].iloc[max_idx[1] + 1]
            time_mask = calculate_window_mask(data, start_date, end_date)
            windowed_data = data[time_mask]
        # update the assertion dictionary
        assertion_dict.update(zip(data.index[time_mask].tolist(), len(data.index[time_mask].tolist()) * [True]))
        # check for aggregation
        if aggregation:
            aggregation_length = '5S' if resample else min(10, int(len(windowed_data.index)))
            timed = True if resample else False
            windowed_data = aggregate_in_windows(windowed_data[selected].copy(deep=True), aggregation_length, timed,
                                                 resample)
            selected = windowed_data.columns.values
            num_of_features = len(selected)
        # and add the new trace
        ints = False if aggregation or 'duration' in old_selected else True
        if windowed_data.shape[0] != 0:     # for the resampling case
            traces += [convert2flexfringe_format(windowed_data[selected], ints)]
        # store also the starting and the ending index of the current time window
        if windowed_data.shape[0] != 0:     # for the resampling case
            traces_indices += [windowed_data.index.tolist()]

    print('Finished with rolling windows!!!')
    # evaluate correctness of the process
    if not all(list(assertion_dict.values())):
        print('There are flows missed -- Check again the implementation!!!')
        print([k for k, v in assertion_dict.items() if not v])
    else:
        print('All flows correctly converted to traces!!!')
    print('Starting writing traces to file...')
    # create the traces' file in the needed format
    f = open(out_filepath, "w")
    f.write(str(len(traces)) + ' ' + '100:' + str(num_of_features) + '\n')
    for trace in traces:
        f.write('1 ' + str(len(trace)) + ' 0:' + ' 0:'.join(trace) + '\n')
    f.close()
    # save also the indices of each trace
    indices_filepath = '.'.join(out_filepath.split('.')[:-1]) + '_indices.pkl'
    with open(indices_filepath, 'wb') as f:
        pickle.dump(traces_indices, f)
    print('Traces written successfully to file!!!')


def parse_dot(dot_path):
    """
    Function for parsing dot files describing multivariate models produced through FlexFringe
    :param dot_path: the path to the dot file
    :return: the parsed model
    """
    with open(dot_path, "r") as f:
        dot_string = f.read()
    # initialize the model
    model = Model()
    # regular expression for parsing a state as well as its contained info
    state_regex = r"(?P<src_state>\d+) \[\s*label=\"(?P<state_info>(.+))\".*\];$"
    # regular expression for parsing a state's contained info
    info_regex = r"((?P<identifier>(fin|symb|attr)+\(\d+\)):\[*(?P<values>(\d|,)+)\]*)+"
    # regular expression for parsing the transitions of a state, as well as the firing conditions
    transition_regex = r"(?P<src_state>.+) -> (?P<dst_state>\d+)( \[label=\"(?P<transition_cond>(.+))\".*\])*;$"
    # regular expression for parsing the conditions firing a transition
    cond_regex = r"((?P<sym>\d+) (?P<ineq_symbol>(<|>|<=|>=|==)) (?P<boundary>\d+))+"
    # boolean flag used for adding states in the model
    cached = False
    for line in re.split(r'\n\t+|\n}', dot_string):  # split the dot file in meaningful lines
        state_matcher = re.match(state_regex, line, re.DOTALL)
        # if a new state is found while parsing
        if state_matcher is not None:
            # check if there is a state pending and, if there is, update the model
            if cached:
                model.add_node(ModelNode(src_state, attributes, fin, tot, dst_nodes, cond_dict))
            src_state = state_matcher.group("src_state")  # find the state number
            # and initialize all its parameters
            attributes = {}
            fin = 0
            tot = 0
            dst_nodes = []
            cond_dict = {}
            # find the state information while parsing
            state_info = state_matcher.group("state_info")
            if state_info != 'root':
                # if the state is not the root one retrieve the state's information
                for info_matcher in re.finditer(info_regex, state_info):
                    identifier = info_matcher.group("identifier")
                    id_values = info_matcher.group("values")
                    if 'fin' in identifier:  # the number of times the state was final
                        fin = int(id_values)
                    elif 'symb' in identifier:  # the number of times the state was visited
                        tot = fin + int(id_values[:-1])
                    else:
                        # the quantiles of each attribute
                        attributes[re.findall(r'\d+', identifier)[0]] = list(map(float, id_values.split(',')[:-1]))
            else:
                # otherwise set the state's name to root (because for some reason 3 labels are used for the root node)
                src_state = 'root'

        transition_matcher = re.match(transition_regex, line, re.DOTALL)
        # if a transition is identified while parsing (should be in the premises of the previously identified state)
        if transition_matcher is not None:
            src_state_1 = transition_matcher.group("src_state")  # the source state number
            if src_state_1 == 'I':  # again the same problem as above
                src_state_1 = 'root'
            # just consistency check that we are still parsing the same state as before
            if src_state != src_state_1:
                print('Something went wrong - Different source states in a state!!!')
                return -1
            # identify the destination state and add it to the list of destinations
            dst_state = transition_matcher.group("dst_state")
            dst_nodes += [dst_state]  # should exist given that transitions come after the identification of a new state
            # check for the transitions' conditions only if the current state is not the root
            if src_state_1 != 'root':
                # find the transition's conditions while parsing
                transition_conditions = transition_matcher.group("transition_cond")
                for condition_matcher in re.finditer(cond_regex, transition_conditions):
                    attribute = condition_matcher.group("sym")  # the attribute contained in the condition
                    inequality_symbol = condition_matcher.group("ineq_symbol")  # the inequality symbol
                    boundary = condition_matcher.group("boundary")  # the numeric limit in the condition
                    # and update the conditions' dictionary
                    # the condition dictonary should be initialized from the state identification stage
                    if dst_state not in cond_dict.keys():
                        cond_dict[dst_state] = []
                    cond_dict[dst_state] += [[attribute, inequality_symbol, float(boundary)]]
            # set the cached flag to True after the first state is fully identified
            cached = True

    # one more node addition for the last state in the file to be added
    model.add_node(ModelNode(src_state, attributes, fin, tot, dst_nodes, cond_dict))
    return model


def traces2list(traces_path):
    """
    Function for converting the trace file into a list of traces ready for further processing
    :param traces_path: the filepath of the traces
    :return: the list of the traces as a list of lists (each trace) of lists (each record in each trace)
    """
    traces = []
    with open(traces_path, "r") as fp:
        # skip first line
        line = fp.readline()
        while line:
            line = fp.readline()
            if line != '':
                # split lines by spaces
                tokens = line.split()
                # gather the records of each trace and keep only the record values and map them to float
                traces += [[list(map(float, t.split(':')[1].split(','))) for t in tokens[2:]]]
    return traces


def run_traces_on_model(traces_path, indices_path, model, attribute_type='train'):
    """
    Function for running a trace file on the provided model and storing the observed attributes' values on it
    :param traces_path: the filepath to the traces' file
    :param indices_path: the filepath to the traces' incices file
    :param model: the given model
    :param attribute_type: the type of the input traces ('train' | 'test')
    :return: the updated model
    """
    traces = traces2list(traces_path)
    with open(indices_path, 'rb') as f:
        traces_indices = pickle.load(f)
    for trace, inds in zip(traces, traces_indices):
        # first fire the transition from root node
        label = model.fire_transition('root', dict())
        for record, ind in zip(trace, inds):
            observed = dict(zip([str(i) for i in range(len(record))], record))
            model.update_attributes(label, observed, attribute_type)
            model.update_indices(label, ind, attribute_type)
            label = model.fire_transition(label, observed)
    return model


def dict2list(d):
    """
    Function for converting a dictionary to list by using the keys of the dictionary as the indices of the list and its
    values as the values of the list.
    :param d: the input dictionary
    :return: the output list
    """
    l = [0] * len(d.keys())
    for k in d.keys():
        l[k] = d[k]
    return l