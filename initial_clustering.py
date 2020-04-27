import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def separate_by_connections(data, threshold):
    """
    Function for grouping a dataset of flows according to their source-destination IP pairs and then splitting the data
    to those with long connections (connections with more than threshold flows) and short connections (connections with
    less than threshold flows). The intuition is that connections with too few flows cannot be properly modelled by the
    multivariate version of flexfringe. Thus, they will be dealt in a different way.
    :param data: the NetFlow dataset
    :param threshold: the threshold number of flows to be considered as a separation limit
    :return: the two split datasets according to the number of flows per connection
    """
    connections_cnts = data.groupby(['src_ip', 'dst_ip']).agg(['count']).reset_index()

    big_connections = connections_cnts[connections_cnts[('date', 'count')] > threshold]
    small_connections = connections_cnts[connections_cnts[('date', 'count')] <= threshold]

    big_data = data.loc[(data['src_ip'].isin(big_connections['src_ip'])) &
                        (data['dst_ip'].isin(big_connections['dst_ip']))].sort_values(by='date').reset_index(drop=True)
    small_data = data.loc[(data['src_ip'].isin(small_connections['src_ip'])) &
                          (data['dst_ip'].isin(small_connections['dst_ip']))].sort_values(by='date').reset_index(drop=True)

    return big_data, small_data


def separate_by_ports(data, threshold):
    """
    Function for grouping a dataset of flows by their destination ports. The flows are separated according to the number
    of times specific ports appear. The intuition here is that for connections with small number of flows it is highly
    likely that flows pointing to similar ports will demonstrate also similar
    behaviour even if they are directed to different IPs.
    :param data: the NetFlow dataset
    :param threshold: the threshold number of flows to be considered as a separation limit
    :return: the two split datasets according to the distribution of ports
    """
    ports_cnts = data.groupby(['dst_port']).agg(['count']).reset_index()

    ports_many_flows = ports_cnts[ports_cnts[('date', 'count')] > threshold]
    ports_few_flows = ports_cnts[ports_cnts[('date', 'count')] <= threshold]

    data_with_few_ports = data.loc[data['dst_port'].isin(ports_many_flows['dst_port'])].reset_index(
        drop=True)
    data_with_lots_ports = data.loc[data['dst_port'].isin(ports_few_flows['dst_port'])].reset_index(
        drop=True)

    return data_with_few_ports, data_with_lots_ports


def separate_by_dst_bytes(data):
    """
    Function for separating a dataset of flows by the number of bytes received. The intuition here is that flows with
    zero bytes received could possibly be malicious.
    :param data: the NetFlow dataset
    :return: the two split datasets according to the number of bytes received
    """
    # TODO: add distinction in case the IOT dataset is not used
    zero_dst_data = data[data['resp_ip_bytes'] == 0].reset_index(drop=True)
    non_zero_dst_data = data[data['resp_ip_bytes'] != 0].reset_index(drop=True)
    return zero_dst_data, non_zero_dst_data


def apply_kmeans(x):
    """
    Function for applying k-means clustering to a given set of data. Before clustering, the ELBOW method is applied to
    decide on the number of clusters to be used.
    :param x: the input dataset
    :return: the fitted clusters
    """
    print('----------------------- Finding optimal number of clusters -----------------------')
    Sum_of_squared_distances = []
    for k in range(1, 11):
        if x.shape[0] > 100000:
            km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=10000)
        else:
            km = KMeans(n_clusters=k, random_state=0)
        km = km.fit(x)
        Sum_of_squared_distances.append(km.inertia_)

    plt.figure()
    plt.plot(range(1, 11), Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.grid()
    plt.show()

    # provide the desired number of discretization points according to the ELBOW plot
    num_clusters = int(input('Enter your preferred number of clusters: '))

    if x.shape[0] > 100000:
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=10000).fit(x)
    else:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x)

    return kmeans


if __name__ == '__main__':
    filepath = 'Datasets/IOT23/Malware-Capture-8-1'
    normal = pd.read_pickle(filepath + '/zeek_normal.pkl')
    anomalous = pd.read_pickle(filepath + '/zeek_anomalous.pkl')
    all_data = pd.concat([normal, anomalous], ignore_index=True).reset_index(drop=True)
    all_data = all_data[all_data['src_ip'] == '192.168.100.113']
    all_data['label'] = (all_data['label'] == 'Malicious').astype(int)

    connections_cnts = all_data.groupby(['src_ip', 'dst_ip']).agg(['count']).reset_index()
    big_connections = connections_cnts[connections_cnts[('date', 'count')] > 100]
    small_connections = connections_cnts[connections_cnts[('date', 'count')] <= 100]

    big_data = all_data.loc[(all_data['src_ip'].isin(big_connections['src_ip'])) &
                             (all_data['dst_ip'].isin(big_connections['dst_ip']))].sort_values(by='date').reset_index(drop=True)

    small_data = all_data.loc[(all_data['src_ip'].isin(small_connections['src_ip'])) &
                            (all_data['dst_ip'].isin(small_connections['dst_ip']))].sort_values(by='date').reset_index(drop=True)

    # Some basic statistics between connections with many flows and short-lived ones

    print('Big connections number of flows')
    print('Benign: ' + str(big_data[big_data['label'] == 0].shape[0]))
    print('Malicious: ' + str(big_data[big_data['label'] == 1].shape[0]))
    print('Small connections number of flows')
    print('Benign: ' + str(small_data[small_data['label'] == 0].shape[0]))
    print('Malicious: ' + str(small_data[small_data['label'] == 1].shape[0]))

    print('Big connections unique destination ports')
    print('Benign: ' + str(big_data[big_data['label'] == 0]['dst_port'].nunique()))
    print('Malicious: ' + str(big_data[big_data['label'] == 1]['dst_port'].nunique()))
    print('Small connections unique destination ports')
    print('Benign: ' + str(small_data[small_data['label'] == 0]['dst_port'].nunique()))
    print('Malicious: ' + str(small_data[small_data['label'] == 1]['dst_port'].nunique()))

    print('Big connections mean bytes sent')
    print('Benign: ' + str(big_data[big_data['label'] == 0]['orig_ip_bytes'].median()))
    print('Malicious: ' + str(big_data[big_data['label'] == 1]['orig_ip_bytes'].median()))
    print('Small connections mean bytes sent')
    print('Benign: ' + str(small_data[small_data['label'] == 0]['orig_ip_bytes'].median()))
    print('Malicious: ' + str(small_data[small_data['label'] == 1]['orig_ip_bytes'].median()))

    print('Big connections mean bytes sent')
    print('Benign: ' + str(big_data[big_data['label'] == 0]['resp_ip_bytes'].median()))
    print('Malicious: ' + str(big_data[big_data['label'] == 1]['resp_ip_bytes'].median()))
    print('Small connections mean bytes sent')
    print('Benign: ' + str(small_data[small_data['label'] == 0]['resp_ip_bytes'].median()))
    print('Malicious: ' + str(small_data[small_data['label'] == 1]['resp_ip_bytes'].median()))

    # First check separability in long-lived and short-lived connections with all features considered
    num_clusters = 5

    x_big = big_data[['protocol_num', 'dst_port', 'orig_ip_bytes', 'resp_ip_bytes']].values
    y_big = big_data['label'].values
    y_det_big = big_data['detailed_label'].values

    if x_big.shape[0] > 100000:
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=10000).fit(x_big)
    else:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x_big)

    print('\n//////////////----------- Results for big connections -----------//////////////')
    for i in range(num_clusters):
        real_cluster_labels = Counter(y_big[kmeans.labels_ == i])
        real_cluster_labels_detailed = '-'.join(list(set(y_det_big[kmeans.labels_ == i])))
        print('------- Cluster {} distribution -------'.format(i))
        print('Benign: ' + str(real_cluster_labels[0]))
        print('Malicious: ' + str(real_cluster_labels[1]))
        print('Types of labels: ' + real_cluster_labels_detailed)

    x_small = small_data[['protocol_num', 'dst_port', 'orig_ip_bytes', 'resp_ip_bytes']].values
    y_small = small_data['label'].values
    y_det_small = small_data['detailed_label'].values

    if x_small.shape[0] > 100000:
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=10000).fit(x_small)
    else:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x_small)

    print('\n//////////////----------- Results for small connections -----------//////////////')
    for i in range(num_clusters):
        real_cluster_labels = Counter(y_small[kmeans.labels_ == i])
        real_cluster_labels_detailed = '-'.join(list(set(y_det_small[kmeans.labels_ == i])))
        print('------- Cluster {} distribution -------'.format(i))
        print('Benign: ' + str(real_cluster_labels[0]))
        print('Malicious: ' + str(real_cluster_labels[1]))
        print('Types of labels: ' + real_cluster_labels_detailed)

    # subsequently check the distribution of destination ports in small connections for a potential port scan
    # and separate between connections with few ports appearing many times and connections with many ports appearing
    # few times
    dst_ports_groups = small_data.groupby(['dst_port']).agg(['count']).reset_index()
    big_dst_ports_groups = dst_ports_groups[dst_ports_groups[('date', 'count')] > 100]
    small_dst_ports_groups = dst_ports_groups[dst_ports_groups[('date', 'count')] <= 100]

    small_big_port_data = small_data.loc[small_data['dst_port'].isin(big_dst_ports_groups['dst_port'])].reset_index(drop=True)
    small_small_port_data = small_data.loc[small_data['dst_port'].isin(small_dst_ports_groups['dst_port'])].reset_index(drop=True)

    num_clusters = 4

    x_big = small_big_port_data[['protocol_num', 'src_port', 'dst_port', 'orig_ip_bytes', 'resp_ip_bytes']].values
    y_big = small_big_port_data['label'].values
    y_det_big = small_big_port_data['detailed_label'].values

    if x_big.shape[0] > 100000:
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=10000).fit(x_big)
    else:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x_big)

    print('\n//////////////----------- Results for small connections with same ports used many times -----------//////////////')
    for i in range(num_clusters):
        real_cluster_labels = Counter(y_big[kmeans.labels_ == i])
        real_cluster_labels_detailed = '-'.join(list(set(y_det_big[kmeans.labels_ == i])))
        print('------- Cluster {} distribution -------'.format(i))
        print('Benign: ' + str(real_cluster_labels[0]))
        print('Malicious: ' + str(real_cluster_labels[1]))
        print('Types of labels: ' + real_cluster_labels_detailed)

    x_small = small_small_port_data[['protocol_num', 'src_port', 'dst_port', 'orig_ip_bytes', 'resp_ip_bytes']].values
    y_small = small_small_port_data['label'].values
    y_det_small = small_small_port_data['detailed_label'].values

    if x_small.shape[0] > 100000:
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=10000).fit(x_small)
    else:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x_small)

    print('\n//////////////----------- Results for small connections with different ports used few times -----------//////////////')
    for i in range(num_clusters):
        real_cluster_labels = Counter(y_small[kmeans.labels_ == i])
        real_cluster_labels_detailed = '-'.join(list(set(y_det_small[kmeans.labels_ == i])))
        print('------- Cluster {} distribution -------'.format(i))
        print('Benign: ' + str(real_cluster_labels[0]))
        print('Malicious: ' + str(real_cluster_labels[1]))
        print('Types of labels: ' + real_cluster_labels_detailed)

    # subsequently separate small connection flows based on their received bytes since flows with no received bytes
    # could be malicious
    small_zero_resp_data = small_data[small_data['resp_ip_bytes'] == 0].reset_index(
        drop=True)
    small_pos_resp_data = small_data[small_data['resp_ip_bytes'] != 0].reset_index(
        drop=True)

    num_clusters = 3

    x_big = small_zero_resp_data[['protocol_num', 'dst_port', 'orig_ip_bytes']].values
    y_big = small_zero_resp_data['label'].values
    y_det_big = small_zero_resp_data['detailed_label'].values

    if x_big.shape[0] > 100000:
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=10000).fit(x_big)
    else:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x_big)

    print(
        '\n//////////////----------- Results for small connections with zero received bytes -----------//////////////')
    for i in range(num_clusters):
        real_cluster_labels = Counter(y_big[kmeans.labels_ == i])
        real_cluster_labels_detailed = '-'.join(list(set(y_det_big[kmeans.labels_ == i])))
        print('------- Cluster {} distribution -------'.format(i))
        print('Benign: ' + str(real_cluster_labels[0]))
        print('Malicious: ' + str(real_cluster_labels[1]))
        print('Types of labels: ' + real_cluster_labels_detailed)

    x_small = small_pos_resp_data[['protocol_num', 'dst_port', 'orig_ip_bytes', 'resp_ip_bytes']].values
    y_small = small_pos_resp_data['label'].values
    y_det_small = small_pos_resp_data['detailed_label'].values

    if x_small.shape[0] > 100000:
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=10000).fit(x_small)
    else:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x_small)

    print(
        '\n//////////////----------- Results for small connections with some received bytes -----------//////////////')
    for i in range(num_clusters):
        real_cluster_labels = Counter(y_small[kmeans.labels_ == i])
        real_cluster_labels_detailed = '-'.join(list(set(y_det_small[kmeans.labels_ == i])))
        print('------- Cluster {} distribution -------'.format(i))
        print('Benign: ' + str(real_cluster_labels[0]))
        print('Malicious: ' + str(real_cluster_labels[1]))
        print('Types of labels: ' + real_cluster_labels_detailed)

    # old unused code -- to be removed eventually
    # all_data['src_ip_num'] = pd.Categorical(all_data['src_ip'], categories=all_data['src_ip'].unique()).codes
    # all_data['dst_ip_num'] = pd.Categorical(all_data['dst_ip'], categories=all_data['dst_ip'].unique()).codes
    #
    # if all_data.shape[0] < 1000:
    #     grouped = all_data.groupby(['src_ip_num', 'dst_ip_num']).agg(
    #         dst_port_median=('dst_port', pd.Series.median),
    #         protocol_num_median=('protocol_num', pd.Series.median),
    #         orig_ip_bytes_min=('orig_ip_bytes', min),
    #         orig_ip_bytes_median=('orig_ip_bytes', pd.Series.median),
    #         orig_ip_bytes_max=('orig_ip_bytes', max),
    #         resp_ip_bytes_min=('resp_ip_bytes', min),
    #         resp_ip_bytes_median=('resp_ip_bytes', pd.Series.median),
    #         resp_ip_bytes_max=('resp_ip_bytes', max),
    #         count=('label', pd.Series.count),
    #         label_num=('label', max),       # currently considering anomaly if even one flow is labelled so
    #         date_max=('date', max),
    #         date_min=('date', min)
    #     )
    #
    #     grouped['flows_per_sec'] = grouped['count'] / (grouped['date_max'] - grouped['date_min']).dt.total_seconds()
    #     grouped['flows_per_sec'] = grouped['flows_per_sec'].replace(np.inf, grouped['count'])
    #
    #     y = grouped['label_num'].values
    #     grouped.drop(columns=['count', 'label_num', 'date_max', 'date_min'], inplace=True)
    #     x = grouped.values