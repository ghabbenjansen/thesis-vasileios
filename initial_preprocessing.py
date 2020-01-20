import pandas as pd


def preprocess_data(filepath):
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
                elements += [columns[ind-1] + ' ' + columns[ind]]
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


def read_data(filepath, flag='CTU-uni', preprocessing=False, background=True, expl=False):
    """
    Helper function to read the datasets into a Pandas dataframe
    :param filepath: the relative path of the file to be read
    :param flag: flag showing the origin of the dataset (CTU or CICIDS)
    :param preprocessing: flag showing if the data need the unidirectional preprocessing
    :param background: flag showing if the background data should be removed
    :param expl: flag regarding the visualization of the error lines in the dataset
    :return: the dataframe with the data
    """
    # if the dataset needs the unidirectional preprocessing
    if preprocessing:
        preprocess_data(filepath)
        filepath += '_preprocessed'

    # parse the datetime field appropriately
    if flag == 'CTU-uni':
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
        parse_field = 'date'
    elif flag in ['CTU-bi', 'CTU-mixed']:
        dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d %H:%M:%S.%f')
        parse_field = 'StartTime'
    else:
        dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H:%M')
        parse_field = 'Timestamp'

    # a simple try-except loop to catch any tokenizing errors in the data (e.g. the FILTER_LEGITIMATE field in the
    # unidirectional flows - for now these lines are ignored) in case the explanatory flag is True
    line = []
    cont = True
    data = []
    while cont:
        try:
            # read the data into a dataframe according to the background flag
            data = pd.read_csv(filepath, delimiter=',', parse_dates=[parse_field], date_parser=dateparse,
                               error_bad_lines=expl, skiprows=line) \
                if background else pd.concat(remove_background(chunk) for chunk in pd.read_csv(filepath,
                                                                                               chunksize=100000,
                                                                                               delimiter=',',
                                                                                               parse_dates=[parse_field],
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
    filepath = 'Datasets/CTU13/scenario6/scenario06_ctu13.pcap.netflow.labeled'
    # filepath = input("Enter the desired filepath: ")
    # Choose between the flags CTU-uni | CTU-bi | CTU-mixed
    flag = 'CTU-uni'
    # flag = input("Enter the desired flag (CTU-uni | CTU-bi | CTU-mixed): ")
    # only for the CTU-mixed case
    given_dates = ['2015/07/26 14:41:51.734831', '2015/07/27 15:51:12.978465']
    # to get preprocessing, necessary for unidirectional netflows, done, set the 'preprocessing' flag to True
    if flag != 'CTU-uni':
        data = read_data(filepath, flag=flag)
    else:
        # data = read_data(filepath, flag=flag, preprocessing=bool(input("Enable preprocessing?: ")))
        data = read_data(filepath, flag=flag, preprocessing=True)
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
        data.dropna(inplace=True)   # drop NaN rows (mostly NaN ports)

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

        # separate the types of features in the dataset
        continuous_features = ['duration', 'src_port', 'dst_port', 'protocol_num', 'flags_num', 'tos', 'packets',
                               'bytes']
        categorical_features = ['protocol', 'flags']

        # check statistics for the most discriminative features in the dataset
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print('---------------- Stats for anomalous flows ----------------')
            print(anomalous[continuous_features].describe())
            print('---------------- Stats for normal flows ----------------')
            print(normal[continuous_features].describe())
            print('---------------- Stats for background flows ----------------')
            print(background[continuous_features].describe())
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
            mask = (data['date'] >= given_dates[0]) & (data['date'] <= given_dates[1]) if len(given_dates) == 2 else \
                data['date'] >= given_dates[0]
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

        # separate the types of features in the dataset
        continuous_features = ['duration', 'Sport', 'Dport' 'protocol_num', 'state_num', 'sTos', 'dTos', 'packets',
                               'bytes', 'src_bytes']
        categorical_features = ['Proto', 'State']

        # check statistics for the most discriminative features in the dataset
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print('---------------- Stats for anomalous flows ----------------')
            print(anomalous[continuous_features].describe())
            print('---------------- Stats for normal flows ----------------')
            print(normal[continuous_features].describe())
            if flag != 'CTU-mixed':
                print('---------------- Stats for background flows ----------------')
                print(background[continuous_features].describe())
    else:
        # TODO: Decide if i will use any of the Canadian datasets
        pass

