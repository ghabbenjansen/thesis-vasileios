import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


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