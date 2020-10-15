#!/usr/bin/python

import pandas as pd
import socket
from os import path
import pickle


def preprocess_unidirectional_data(filepath):
    """
    Helper function for preprocessing the unidirectional netflows. The ips should be separated from the ports, the date
    values should be taken care of while splitting, while the separator should be converted from space to comma to meet
    the specifications of the rest datasets
    :param filepath: the relative path of the file to be processed
    :return: a file with the preprocessed data is created
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    fout = open(filepath + '_preprocessed', 'w')

    column_names = ['date', 'duration', 'protocol', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'flags', 'tos',
                    'packets', 'bytes', 'flows', 'label']

    fout.write(','.join(column_names) + '\n')
    for i, line in enumerate(lines[1:]):
        elements = []
        columns = line.split()
        for ind in range(len(columns)):
            # take into account that date has a space
            if ind == 1:
                elements += [columns[ind - 1] + ' ' + columns[ind]]
            # split source ips and ports
            elif ind == 4:
                elements += [columns[ind].split(':')[0]]
                elements += ['NaN' if len(columns[ind].split(':')) == 1 or not columns[ind].split(':')[1].isdigit()
                             else columns[ind].split(':')[1]]
            # split destination ips and ports
            elif ind == 6:
                elements += [columns[ind].split(':')[0]]
                elements += ['NaN' if len(columns[ind].split(':')) == 1 or not columns[ind].split(':')[1].isdigit()
                             else columns[ind].split(':')[1]]
            # ignore these two indexes
            elif ind == 0 or ind == 5:
                pass
            else:
                elements += [columns[ind]]

        fout.write(','.join(elements) + '\n')
        if i % 10000:
            print(str(i) + ' lines have been processed...')
    fout.close()


def read_data(filepath, flag='CTU-uni', preprocessing=None, chunks=False, expl=False):
    """
    Helper function to read the datasets into a Pandas dataframe
    :param filepath: the relative path of the file to be read
    :param flag: flag showing the origin of the dataset (CTU | CICIDS | USNW)
    :param preprocessing: flag only applicable to the unidirectional Netflow case of CTU-13
    :param chunks: flag showing if the input data should be parsed in chunks into the dataframe
    :param expl: flag regarding the visualization of the error lines in the dataset
    :return: the dataframe with the data
    """
    # if the dataset needs some preprocessing
    if preprocessing:
        preprocess_unidirectional_data(filepath)
        filepath += '_preprocessed'

    # Set the flags for dataframe parsing in the appropriate way for each dataset
    # The values set are the following:
    # delimeter: Delimiter to use to separate the values in each column
    # header: row number to use as header - if it is set to None then no header is inferred
    # names: List of column names to use, in case header is set to None
    # usecols: The set of columns to be used
    # skiprows: The row numbers to skip or the number of rows to skip from the beginning of the document
    # sikpfooter: Number of lines at bottom of file to skip
    # na_values: Additional strings to recognize as NA/NaN
    # parse_field: The columns to parse with the dateparse function
    # dateparse: Function to use for converting a sequence of string columns to an array of datetime instances
    delimiter = ','
    header = None
    skiprows = 1
    skipfooter = 0
    na_values = []
    parse_field = ['date']
    encoding = 'utf_8'
    engine = 'python'

    # Unidirectional Netflow data from CTU-13 dataset
    if flag == 'CTU-uni':
        names = ['date', 'duration', 'protocol', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'flags', 'packets',
                 'bytes', 'label']
        usecols = [_ for _ in range(0, 8)] + [9, 10, 12]
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    # Bidirectional Netflow data from CTU-13 dataset
    elif flag == 'CTU-bi':
        names = ['date', 'duration', 'protocol', 'src_ip', 'src_port', 'direction', 'dst_ip', 'dst_port', 'state',
                 'packets', 'bytes', 'src_bytes', 'label']
        usecols = [_ for _ in range(0, 9)] + [_ for _ in range(11, 15)]
        dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d %H:%M:%S.%f')
    # Netflow data from UNSW-NB15 dataset
    elif flag == 'UNSW':
        names = ['src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol', 'state', 'duration', 'src_bytes',
                 'dst_bytes', 'missed_src_bytes', 'missed_dst_bytes', 'service', 'src_packets', 'dst_packets',
                 'date', 'end_date', 'detailed_label', 'label']
        usecols = [_ for _ in range(0, 9)] + [_ for _ in range(11, 14)] + [16, 17, 28, 29, 47, 48]
        na_values = ['-']
        skiprows = []
        dateparse = lambda x: pd.to_datetime(x, unit='s')
        parse_field = ['date', 'end_date']
        # special handling for the first dataset of the UNSW datasets
        if '1' in filepath.split('/')[2]:
            skiprows = 1
    # Netflow data from CICIDS2017 dataset
    else:
        # in this dataset the features 'total length of forward packets' and 'total length of
        # backward packets' have been considered as sent and received bytes
        names = ['src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol', 'date', 'duration', 'total_fwd_packets',
                 'total_bwd_packets', 'src_bytes', 'dst_bytes', 'label']
        usecols = [_ for _ in range(1, 12)] + [84]
        dateparse = lambda x: pd.to_datetime(x, dayfirst=True)
        encoding = 'latin_1'

    # a simple try-except loop to catch any tokenizing errors in the data (e.g. the FILTER_LEGITIMATE field in the
    # unidirectional flows - for now these lines are ignored) in case the explanatory flag is True
    cont = True
    data = []
    while cont:
        try:
            # read the data into a dataframe according to the background flag
            data = pd.read_csv(filepath, delimiter=delimiter, header=header, names=names, parse_dates=parse_field,
                               date_parser=dateparse, usecols=usecols, na_values=na_values, error_bad_lines=expl,
                               engine=engine, skiprows=skiprows, skipfooter=skipfooter, encoding=encoding) if not chunks \
                else pd.concat(pd.read_csv(filepath, chunksize=1000000, delimiter=delimiter, header=header, names=names,
                                           parse_dates=parse_field, date_parser=dateparse, usecols=usecols,
                                           na_values=na_values, error_bad_lines=expl, engine=engine, skiprows=skiprows,
                                           encoding=encoding))
            cont = False
        except Exception as e:
            errortype = str(e).split('.')[0].strip()
            if errortype == 'Error tokenizing data':
                cerror = str(e).split(':')[1].strip().replace(',', '')
                nums = [n for n in cerror.split(' ') if str.isdigit(n)]
                skiprows.append(int(nums[1]) - 1)
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                err_line = lines[int(nums[1]) - 1]
                print(err_line)
            else:
                print(errortype)

    return data


if __name__ == '__main__':
    filepath = input("Enter the desired filepath: ")

    # Choose between the flags CTU-uni | CTU-bi | CTU-mixed | CICIDS | UNSW
    while True:
        flag = input("Enter the desired flag (CTU-uni | CTU-bi | CICIDS | UNSW): ")
        if flag in ['CTU-uni', 'CTU-bi', 'CICIDS', 'UNSW']:
            break

    # flag for modifications in preprocessing for state-of-the-art experiments
    sota = True

    print('Reading data from ' + filepath + '...\n')
    # to get preprocessing, necessary for unidirectional netflows, done, set the 'preprocessing' flag to True
    if flag == 'CTU-uni':
        data = read_data(filepath, flag=flag, preprocessing='uni' if bool(input("Enable preprocessing (for NO give no answer)? ")) else None)
    else:
        data = read_data(filepath, flag=flag, chunks=True)

    print('Dataset from ' + filepath + ' has been successfully read!!!\n')
    print('Starting initial preprocessing...\n')

    # resetting indices for data
    data = data.reset_index(drop=True)

    # some more preprocessing on the specific fields of the dataframe
    # Unidirectional flows of CTU-13 dataset
    if flag == 'CTU-uni':
        # parse packets, and bytes as integers instead of strings
        data['packets'] = data['packets'].astype(int)
        data['bytes'] = data['bytes'].astype(int)

        # parse duration as float
        data['duration'] = data['duration'].astype(float)

        # add the numerical representation of the categorical data
        data['protocol_num'] = pd.Categorical(data['protocol'], categories=data['protocol'].unique()).codes
        data['flags_num'] = pd.Categorical(data['flags'], categories=data['flags'].unique()).codes

        # handle NaN values (mostly NaN ports)
        # data.dropna(inplace=True) # one solution would be to drop the flows
        data['src_port'].fillna('-1', inplace=True)
        data['dst_port'].fillna('-1', inplace=True)

        # since NaN values have been removed from ports
        data['src_port'] = data['src_port'].astype(int)
        data['dst_port'] = data['dst_port'].astype(int)

        # split the data according to their labels and sort them by date
        anomalous = data[data['label'] == 'Botnet']
        anomalous = anomalous.reset_index(drop=True)
        anomalous.sort_values(by=['date'], inplace=True)

        normal = data[data['label'] == 'LEGITIMATE']
        normal = normal.reset_index(drop=True)
        normal.sort_values(by=['date'], inplace=True)

        background = data[data['label'] == 'Background']
        background = background.reset_index(drop=True)
        background.sort_values(by=['date'], inplace=True)

        # save the separated data
        anomalous.to_pickle('/'.join(filepath.split('/')[0:3]) + '/netflow_anomalous.pkl')
        normal.to_pickle('/'.join(filepath.split('/')[0:3]) + '/netflow_normal.pkl')
        background.to_pickle('/'.join(filepath.split('/')[0:3]) + '/netflow_background.pkl')
    # Bidirectional flows of CTU-13 dataset
    elif flag == 'CTU-bi':
        # the background data are taken into account only in state-of-the-art experiments
        if not sota:
            data = data[~data['label'].str.contains("Background")]
        else:
            # only hosts from the host network with more than 150 flows in total are kept
            data = data[data['src_ip'].str.contains("147.32.8")]
            hosts = data.groupby('src_ip').size()
            ips = hosts[hosts > 150].index
            data = data[data['src_ip'].isin(ips)]

        # for now the state and direction features are kept but are ignored in the pipeline
        # parse packets, and bytes as integers instead of strings
        data['packets'] = data['packets'].astype(int)
        data['bytes'] = data['bytes'].astype(int)
        data['src_bytes'] = data['src_bytes'].astype(int)
        data['dst_bytes'] = data['bytes'] - data['src_bytes']

        # parse duration as float
        data['duration'] = data['duration'].astype(float)

        # handle NaN values (mostly NaN ports)
        data['state'].fillna('missing', inplace=True)
        # data.dropna(inplace=True)  # dropping the rows with nan values (nan ports mostly) is a solution
        data['src_port'].fillna('-1', inplace=True)
        data['dst_port'].fillna('-1', inplace=True)

        # add the numerical representation of the categorical data
        protocol_categories = ['udp', 'tcp', 'icmp', 'arp', 'igmp', 'rtp', 'pim', 'rtcp', 'udt', 'ipv6']
        data['protocol_num'] = pd.Categorical(data['protocol'], categories=protocol_categories).codes
        # data['state_num'] = pd.Categorical(data['state'], categories=data['state'].unique()).codes

        # handle special hexadecimal values in the port columns
        data['src_port'] = data['src_port'].apply(lambda x: int(x, 16) if 'x' in x else x)
        data['dst_port'] = data['dst_port'].apply(lambda x: int(x, 16) if 'x' in x else x)

        # since NaN values have been removed from ports
        data['src_port'] = data['src_port'].astype(int)
        data['dst_port'] = data['dst_port'].astype(int)

        # split the data according to their labels and sort them by date
        anomalous = data[data['label'].str.contains("From-Botnet")]
        anomalous = anomalous.reset_index(drop=True)
        anomalous.sort_values(by=['date'], inplace=True)

        if not sota:
            normal = data[data['label'].str.contains("From-Normal")]
        else:
            normal = data[~data['label'].str.contains("Botnet")]
        normal = normal.reset_index(drop=True)
        normal.sort_values(by=['date'], inplace=True)

        # save the separated data
        anomalous.to_pickle('/'.join(filepath.split('/')[0:3]) + '/binetflow_anomalous.pkl')
        if not sota:
            normal.to_pickle('/'.join(filepath.split('/')[0:3]) + '/binetflow_normal.pkl')
        else:
            normal.to_pickle('/'.join(filepath.split('/')[0:3]) + '/binetflow_normal_sota.pkl')
    # Flows of UNSW-NB15 dataset
    elif flag == 'UNSW':
        # fill appropriately columns that contain too many NaN values
        data['detailed_label'].fillna('missing', inplace=True)
        data['service'].fillna('missing', inplace=True)
        data['src_port'].fillna('-1', inplace=True)
        data['dst_port'].fillna('-1', inplace=True)

        # handle special hexadecimal values in the destination port columns
        if data.src_port.dtype == object:
            data['src_port'] = data['src_port'].apply(lambda x: int(x, 16) if 'x' in x else x)
        if data.dst_port.dtype == object:
            data['dst_port'] = data['dst_port'].apply(lambda x: int(x, 16) if 'x' in x else x)

        # parse packets, bytes, and ports as integers instead of strings
        data['src_port'] = data['src_port'].astype(int)
        data['dst_port'] = data['dst_port'].astype(int)
        data['src_bytes'] = data['src_bytes'].astype(int)
        data['dst_bytes'] = data['dst_bytes'].astype(int)
        data['missed_src_bytes'] = data['missed_src_bytes'].astype(int)
        data['missed_dst_bytes'] = data['missed_dst_bytes'].astype(int)
        data['src_packets'] = data['src_packets'].astype(int)
        data['dst_packets'] = data['dst_packets'].astype(int)

        # parse duration as float
        data['duration'] = data['duration'].astype(float)

        # for now the state and service features are kept but are ignored in the pipeline
        # add the numerical representation of the categorical data
        protocol_names_filepath = '/'.join(filepath.split('/')[0:2]) + '/protocol_names.pkl'
        if path.exists(protocol_names_filepath):
            with open(protocol_names_filepath, 'rb') as f:
                protocol_categories = pickle.load(f)
        else:
            protocol_categories = data.protocol.unique().tolist()
            with open(protocol_names_filepath, 'wb') as f:
                pickle.dump(protocol_categories, f)

        if not set(data.protocol.unique().tolist()).issubset(set(protocol_categories)):
            print('New protocol types found!!!!')
        data['protocol_num'] = pd.Categorical(data['protocol'], categories=protocol_categories).codes
        # data['service_num'] = pd.Categorical(data['service'], categories=data['service'].unique()).codes
        # data['state_num'] = pd.Categorical(data['state'], categories=data['state'].unique()).codes

        # split the data according to their labels and sort them by date
        anomalous = data[data['label'] == 1]
        anomalous = anomalous.reset_index(drop=True)
        anomalous.sort_values(by=['date'], inplace=True)
        anomalous.reset_index(drop=True, inplace=True)

        normal = data[data['label'] == 0]
        normal = normal.reset_index(drop=True)
        normal.sort_values(by=['date'], inplace=True)
        normal.reset_index(drop=True, inplace=True)

        # save the separated data
        anomalous.to_pickle('/'.join(filepath.split('/')[0:2]) + '/' + filepath.split('/')[2].split('.')[0] +
                            '_anomalous.pkl')
        normal.to_pickle('/'.join(filepath.split('/')[0:2]) + '/' + filepath.split('/')[2].split('.')[0] +
                         '_normal.pkl')
    # Flows of CICIDS2017 dataset
    else:
        # drop rows with NaN values (mostly for the Thursday morning dataset)
        data.dropna(inplace=True)

        # parse packets, bytes, and ports as integers instead of strings
        data['src_port'] = data['src_port'].astype(int)
        data['dst_port'] = data['dst_port'].astype(int)
        data['total_fwd_packets'] = data['total_fwd_packets'].astype(int)
        data['total_bwd_packets'] = data['total_bwd_packets'].astype(int)
        data['src_bytes'] = data['src_bytes'].astype(int)
        data['dst_bytes'] = data['dst_bytes'].astype(int)
        data['protocol_num'] = data['protocol'].astype(int)

        # convert the numerical protocol values to strings according to their code
        table = {num: name[8:] for name, num in vars(socket).items() if name.startswith("IPPROTO")}
        data['protocol'] = data['protocol'].apply(lambda x: table[x])

        # parse duration as float
        data['duration'] = data['duration'].astype(float)

        # split the data according to their labels and sort them by date
        anomalous = data[data['label'] != 'BENIGN']
        anomalous = anomalous.reset_index(drop=True)
        anomalous.sort_values(by=['date'], inplace=True)
        anomalous.reset_index(drop=True, inplace=True)

        normal = data[data['label'] == 'BENIGN']
        normal = normal.reset_index(drop=True)
        normal.sort_values(by=['date'], inplace=True)
        normal.reset_index(drop=True, inplace=True)

        # save the separated data
        anomalous.to_pickle('/'.join(filepath.split('/')[0:3]) + '/anomalous.pkl')
        normal.to_pickle('/'.join(filepath.split('/')[0:3]) + '/normal.pkl')
    print('Data preprocessed and split by label!!!')
