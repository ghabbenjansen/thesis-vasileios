import pandas as pd
from pandas.tseries.offsets import DateOffset


def convert2flexfringe_format(win_data):
    return list(map(lambda x: ','.join(x), win_data.to_numpy().tolist()))


def extract_traces(in_filepath, out_filepath, window, stride, selected):
    data = pd.read_pickle(in_filepath)
    start_date = data['date'].iloc[0]
    end_date = start_date + DateOffset(seconds=window)
    traces = []
    while end_date < data['date'].iloc[-1]:
        time_mask = (data['date'] >= start_date) & (data['date'] <= end_date)
        windowed_data = data[time_mask]     # TODO: not sure about its validity - To check it
        traces += [convert2flexfringe_format(windowed_data[selected])]
        start_date += DateOffset(seconds=stride)
        end_date = start_date + DateOffset(seconds=window)

    f = open(out_filepath + "training_traces.txt", "w")
    f.write(str(len(traces)) + ' ' + '100:' + str(len(selected)))
    for trace in traces:
        f.write('1 ' + str(len(trace)) + ' 0:' + ' 0:'.join(trace))
    f.close()

