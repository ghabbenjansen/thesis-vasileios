import pandas as pd


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

    fout.write(','.join(column_names))
    fout.write('\n')
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

        fout.write(','.join(elements))
        fout.write('\n')

        if i % 10000:
            print(str(i) + ' lines have been processed...')
    fout.close()


def preprocess_zeek_data(filepath):
    """
    Helper function to preprocess the Zeek flows of the IOT dataset. It mainly sets the column names, removes the uid
    column, and replaces '-' values with NaN.
    :param filepath: the relative path of the file to be processed
    :return: a file with the preprocessed data is created
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    fout = open('.'.join(filepath.split('.')[:-1]) + '.preprocessed' + '.txt', 'w')

    column_names = ['timestamp', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol', 'service', 'duration',
                    'orig_bytes', 'resp_bytes', 'state', 'local_orig', 'local_resp', 'missed_bytes', 'history',
                    'orig_packets', 'orig_ip_bytes', 'resp_packets', 'resp_ip_bytes', 'tunnel_parents', 'label',
                    'detailed_label']

    fout.write(','.join(column_names))
    fout.write('\n')
    for i, line in enumerate(lines[8:-1]):
        elements = []
        columns = line.split('\t')
        for ind in range(len(columns)):
            # ignore the unique id column
            if ind == 1:
                pass
            # otherwise just put NaN value to the '-' cells
            elif ind == len(columns) - 1:
                elements += ['NaN' if c == '-' else c for c in columns[ind].split()]  # specific handling
            else:
                elements += ['NaN' if columns[ind] == '-' else columns[ind]]
        fout.write(','.join(elements))
        fout.write('\n')

        if i % 10000:
            print(str(i) + ' lines have been processed...')
    fout.close()


def read_data(filepath, flag='CTU-uni', preprocessing=None, background=True, expl=False):
    """
    Helper function to read the datasets into a Pandas dataframe
    :param filepath: the relative path of the file to be read
    :param flag: flag showing the origin of the dataset (CTU | CICIDS | CIDDS | IOT | USNW)
    :param preprocessing: flag showing if the data need any kind of preprocessing
    :param background: flag showing if the background data should be removed (for the CTU-13 dataset mostly)
    :param expl: flag regarding the visualization of the error lines in the dataset
    :return: the dataframe with the data
    """
    # if the dataset needs some preprocessing
    if preprocessing is not None:
        if preprocessing == 'uni':
            preprocess_unidirectional_data(filepath)
            filepath += '_preprocessed'
        elif preprocessing == 'zeek':
            preprocess_zeek_data(filepath)
            filepath = '.'.join(filepath.split('.')[:-1]) + '.preprocessed' + '.txt'

    # TODO: fix the flags for every dataset so that no preprocessing is needed
    # parse the datetime field appropriately
    if flag == 'CTU-uni':
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
        parse_field = ['date']
        header = 0
    elif flag in ['CTU-bi', 'CTU-mixed']:
        dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d %H:%M:%S.%f')
        parse_field = ['StartTime']
        header = 0
    elif flag == 'IOT':
        dateparse = lambda x: pd.to_datetime(x, unit='s')
        parse_field = ['timestamp']
        header = 0
    elif flag == 'UNSW':
        dateparse = lambda x: pd.to_datetime(x, unit='s')
        parse_field = ['start_time', 'end_time']
        usecols = [_ for _ in range(0, 9)] + [_ for _ in range(11, 14)] + [16, 17, 28, 29, 47, 48]
        na_values = ['-']
        header = None
        names = ['src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol', 'state', 'duration', 'src_bytes',
                 'dst_bytes', 'missed_src_bytes', 'missed_dst_bytes', 'service', 'src_packets', 'dst_packets',
                 'start_time', 'end_time', 'detailed_label', 'label']
    elif flag == 'CICIDS':
        dateparse = lambda x: pd.to_datetime(x, dayfirst=True)
        parse_field = ['Timestamp']
        header = 0
    else:
        dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H:%M')
        parse_field = ['Timestamp']
        header = 0

    # a simple try-except loop to catch any tokenizing errors in the data (e.g. the FILTER_LEGITIMATE field in the
    # unidirectional flows - for now these lines are ignored) in case the explanatory flag is True
    line = []
    cont = True
    data = []
    while cont:
        try:
            # read the data into a dataframe according to the background flag
            data = pd.read_csv(filepath, delimiter=',', header=header, names=names, parse_dates=parse_field,
                               date_parser=dateparse, usecols=usecols, na_values=na_values, error_bad_lines=expl,
                               skiprows=line) if background else pd.concat(remove_background(chunk) for chunk in
                                                                           pd.read_csv(filepath, chunksize=100000,
                                                                                       delimiter=',',
                                                                                       parse_dates=parse_field,
                                                                                       date_parser=dateparse,
                                                                                       error_bad_lines=expl,
                                                                                       skiprows=line))
            cont = False
        except Exception as e:
            errortype = str(e).split('.')[0].strip()

            if errortype == 'Error tokenizing data':
                cerror = str(e).split(':')[1].strip().replace(',', '')
                nums = [n for n in cerror.split(' ') if str.isdigit(n)]
                line.append(int(nums[1]) - 1)
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                err_line = lines[int(nums[1]) - 1]
                print(err_line)
            else:
                print(errortype)

    if not background:
        data.to_pickle(filepath + '_no_background.pkl')
    return data


def remove_background(df):
    """
    Helper function removing background flows from a given dataframe
    :param df: the dataframe
    :return: the no-background dataframe
    """
    df = df[df['label'] != 'Background']
    return df


if __name__ == '__main__':
    filepath = 'Datasets/UNSW-NB15/UNSW-NB15_1.csv'
    # filepath = input("Enter the desired filepath: ")
    # Choose between the flags CTU-uni | CTU-bi | CTU-mixed | CICIDS | CIDDS | UNSW | IOT
    flag = 'UNSW'
    # while True:
    #     flag = input("Enter the desired flag (CTU-uni | CTU-bi | CTU-mixed | CICIDS | CIDDS | UNSW | IOT): ")
    #     if flag in ['CTU-uni', 'CTU-bi', 'CTU-mixed', 'CICIDS', 'CIDDS', 'UNSW', 'IOT']:
    #         break
    # only for the CTU-mixed case
    given_dates = ['2015/07/26 14:41:51.734831', '2015/07/27 15:51:12.978465']
    # to get preprocessing, necessary for unidirectional netflows, done, set the 'preprocessing' flag to True
    if flag == 'CTU-uni':
        data = read_data(filepath, flag=flag, preprocessing='uni' if bool(input("Enable preprocessing (for NO give no "
                                                                                "answer)? ")) else None)
    elif flag == 'IOT':
        data = read_data(filepath, flag=flag, preprocessing='zeek' if bool(input("Enable preprocessing (for NO give no "
                                                                                 "answer)? ")) else None)
    else:
        data = read_data(filepath, flag=flag)

    print('Dataset from ' + filepath + ' has been successfully read!!!')
    # resetting indices for data
    data = data.reset_index(drop=True)

    # some more preprocessing on the specific fields of the dataframe
    if flag == 'CTU-uni':
        # parse packets, bytes, and ToS as integers instead of strings
        data['packets'] = data['packets'].astype(int)
        data['bytes'] = data['bytes'].astype(int)
        data['tos'] = data['tos'].astype(int)

        # parse duration as float
        data['duration'] = data['duration'].astype(float)

        # add the numerical representation of the categorical data
        data['protocol_num'] = pd.Categorical(data['protocol'], categories=data['protocol'].unique()).codes
        data['flags_num'] = pd.Categorical(data['flags'], categories=data['flags'].unique()).codes

        # drop the flows column since it is meaningless
        data.drop(columns=['flows'], inplace=True)
        data.dropna(inplace=True)  # drop NaN rows (mostly NaN ports)

        # since NaN values have been removed from ports
        data['src_port'] = data['src_port'].astype(int)
        data['dst_port'] = data['dst_port'].astype(int)

        # split the data according to their labels
        anomalous = data[data['label'] == 'Botnet']
        anomalous = anomalous.reset_index(drop=True)

        normal = data[data['label'] == 'LEGITIMATE']
        normal = normal.reset_index(drop=True)

        background = data[data['label'] == 'Background']
        background = background.reset_index(drop=True)

        # save the separated data
        anomalous.to_pickle('/'.join(filepath.split('/')[0:3]) + '/netflow_anomalous.pkl')
        normal.to_pickle('/'.join(filepath.split('/')[0:3]) + '/netflow_normal.pkl')
        background.to_pickle('/'.join(filepath.split('/')[0:3]) + '/netflow_background.pkl')
    elif flag == 'IOT':
        # TODO: check how to handle NaN values
        # parse packets, bytes, and ports as integers instead of strings
        data['src_port'] = data['src_port'].astype(int)
        data['dst_port'] = data['dst_port'].astype(int)
        data['orig_bytes'] = data['orig_bytes'].astype(int)
        data['resp_bytes'] = data['resp_bytes'].astype(int)
        data['missed_bytes'] = data['missed_bytes'].astype(int)
        data['orig_packets'] = data['orig_packets'].astype(int)
        data['orig_ip_bytes'] = data['orig_ip_bytes'].astype(int)
        data['resp_packets'] = data['resp_packets'].astype(int)
        data['resp_ip_bytes'] = data['resp_ip_bytes'].astype(int)

        # parse duration as float
        data['duration'] = data['duration'].astype(float)

        # add the numerical representation of the categorical data
        data['protocol_num'] = pd.Categorical(data['protocol'], categories=data['protocol'].unique()).codes
        data['service_num'] = pd.Categorical(data['service'], categories=data['service'].unique()).codes
        data['state_num'] = pd.Categorical(data['state'], categories=data['state'].unique()).codes

        # drop the local_orig, local_resp, and history columns since they seem meaningless (NaN values mostly)
        data.drop(columns=['local_orig', 'local_resp', 'history'], inplace=True)

        # split the data according to their labels
        anomalous = data[data['label'] == 'Malicious']
        anomalous = anomalous.reset_index(drop=True)

        normal = data[data['label'] == 'Benign']
        normal = normal.reset_index(drop=True)

        # save the separated data
        anomalous.to_pickle('/'.join(filepath.split('/')[0:3]) + '/zeek_anomalous.pkl')
        normal.to_pickle('/'.join(filepath.split('/')[0:3]) + '/zeek_normal.pkl')
    elif flag in ['CTU-bi', 'CTU-mixed']:
        # parse packets, bytes, and ToS as integers instead of strings
        data['packets'] = data['TotPkts'].astype(int)
        data['bytes'] = data['TotBytes'].astype(int)
        data['src_bytes'] = data['SrcBytes'].astype(int)

        data['sTos'] = data['sTos'].fillna(method='ffill')
        data['sTos'] = data['sTos'].astype(int)
        data['dTos'] = data['dTos'].fillna(method='ffill')
        data['dTos'] = data['dTos'].astype(int)

        # parse duration as float
        data['duration'] = data['Dur'].astype(float)

        # add the numerical representation of the categorical data
        data['protocol_num'] = pd.Categorical(data['Proto'], categories=data['Proto'].unique()).codes
        data['state_num'] = pd.Categorical(data['State'], categories=data['State'].unique()).codes

        # drop meaningless columns
        data.drop(columns=['TotPkts', 'TotBytes', 'SrcBytes', 'Dur'], inplace=True)
        data.dropna(inplace=True)  # drop NaN rows (mostly NaN ports)

        # since NaN values have been removed from ports
        data['Sport'] = data['Sport'].astype(int)
        data['Dport'] = data['Dport'].astype(int)

        # in case of the mixed CTU flows drop also the deep packet data
        if flag == 'CTU-mixed':
            data.drop(columns=['srcUdata', 'dstUdata', 'Label'], inplace=True)
            mask = (data['StartTime'] >= given_dates[0]) & (data['StartTime'] <= given_dates[1]) \
                if len(given_dates) == 2 else data['StartTime'] >= given_dates[0]
            # the rows that agree with the mask are anomalous
            anomalous = data.loc[mask]
            anomalous = anomalous.reset_index(drop=True)

            normal = data.loc[~mask]
            normal = normal.reset_index(drop=True)

            # save the separated data
            anomalous.to_pickle('/'.join(filepath.split('/')[0:3]) + '/binetflow_anomalous.pkl')
            normal.to_pickle('/'.join(filepath.split('/')[0:3]) + '/binetflow_normal.pkl')
        else:
            # split the data according to their labels
            anomalous = data[data['Label'].str.contains("Botnet")]
            anomalous = anomalous.reset_index(drop=True)

            normal = data[data['Label'].str.contains("Normal")]
            normal = normal.reset_index(drop=True)

            background = data[data['Label'].str.contains("Background")]
            background = background.reset_index(drop=True)

            # save the separated data
            anomalous.to_pickle('/'.join(filepath.split('/')[0:3]) + '/binetflow_anomalous.pkl')
            normal.to_pickle('/'.join(filepath.split('/')[0:3]) + '/binetflow_normal.pkl')
            background.to_pickle('/'.join(filepath.split('/')[0:3]) + '/binetflow_background.pkl')

    elif flag == 'CICIDS':
        # TODO: Preprocessing for the CICIDS 2017 dataset
        pass
    elif flag == 'CIDDS':
        # TODO: Preprocessing for the CIDDS dataset
        pass
    else:
        # TODO: check how to handle NaN values
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

        # add the numerical representation of the categorical data
        data['protocol_num'] = pd.Categorical(data['protocol'], categories=data['protocol'].unique()).codes
        data['service_num'] = pd.Categorical(data['service'], categories=data['service'].unique()).codes
        data['state_num'] = pd.Categorical(data['state'], categories=data['state'].unique()).codes

        # split the data according to their labels
        anomalous = data[data['label'] == '1']
        anomalous = anomalous.reset_index(drop=True)

        normal = data[data['label'] == '0']
        normal = normal.reset_index(drop=True)

        # save the separated data
        anomalous.to_pickle('/'.join(filepath.split('/')[0:2]) + '/' + filepath.split('.')[-1] + '_anomalous.pkl')
        normal.to_pickle('/'.join(filepath.split('/')[0:2]) + '/' + filepath.split('.')[-1] + '_normal.pkl')
