import hdbscan
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def print_results(predicted, real, real_spec):
    results_dict = {}
    result_spec_dict = {}
    for i in range(len(predicted)):
        # first check the high level labels
        if predicted[i] not in results_dict.keys():
            results_dict[predicted[i]] = {}
            results_dict[predicted[i]][real[i]] = 1
        else:
            if real[i] not in results_dict[predicted[i]].keys():
                results_dict[predicted[i]][real[i]] = 1
            else:
                results_dict[predicted[i]][real[i]] += 1

        # then check the specific labels
        if predicted[i] not in result_spec_dict.keys():
            result_spec_dict[predicted[i]] = {}
            result_spec_dict[predicted[i]][real_spec[i]] = 1
        else:
            if real_spec[i] not in result_spec_dict[predicted[i]].keys():
                result_spec_dict[predicted[i]][real_spec[i]] = 1
            else:
                result_spec_dict[predicted[i]][real_spec[i]] += 1

    print('Distribution of high level labels in clusters')
    for k, v in results_dict.items():
        print('the distribution of labels in cluster ' + str(k) + 'is:')
        for kk, vv in v.items():
            print(str(v) + ': ' + str(vv))

    print('Distribution of specific labels in clusters')
    for k, v in result_spec_dict.items():
        print('the distribution of labels in cluster ' + str(k) + 'is:')
        for kk, vv in v.items():
            print(str(v) + ': ' + str(vv))


if __name__ == '__main__':
    # filepath_normal, filepath_malicious = input("Enter the desired filepaths (normal anomalous) separated by space: ").\
    #     split()
    filepath_normal = 'Datasets/IOT23/Malware-Capture-34-1/zeek_normal.pkl'
    filepath_malicious = 'Datasets/IOT23/Malware-Capture-34-1/zeek_anomalous.pkl'
    normal = pd.read_pickle(filepath_normal)
    anomalous = pd.read_pickle(filepath_malicious)

    # prepare the data for clustering
    all_data = pd.concat([normal, anomalous], ignore_index=True).sort_values(by='date')
    all_data_grouped_connection = all_data.groupby(['src_ip', 'dst_ip']).agg({'count': 'count'})
    sel_data = all_data.loc[(all_data['src_ip'].isin(
        all_data_grouped_connection.loc[all_data_grouped_connection['count'] > 100]['src_ip'])) &
                 (all_data['dst_ip'].isin(all_data_grouped_connection.loc[all_data_grouped_connection['count'] > 100]
                                          ['dst_ip']))].reset_index(drop=True)
    sel_data['orig_packets_per_s'] = sel_data['orig_packets'] / sel_data['duration']
    sel_data['resp_packets_per_s'] = sel_data['resp_packets'] / sel_data['duration']
    sel_data['orig_bytes_per_s'] = sel_data['orig_ip_bytes'] / sel_data['duration']
    sel_data['resp_bytes_per_s'] = sel_data['resp_ip_bytes'] / sel_data['duration']
    sel_data['label_num'] = pd.Categorical(sel_data['label'], categories=sel_data['label'].unique()).codes
    sel_data['detailed_label_num'] = pd.Categorical(sel_data['detailed_label'],
                                                   categories=sel_data['detailed_label'].unique()).codes
    data = sel_data.drop(columns=['date', 'src_ip', 'dst_ip', 'protocol', 'state', 'missed_bytes', 'label', 'label_num',
                                  'detailed_label', 'detailed_label_num'])
    x = data.values
    y = sel_data['label_num'].values
    y_spec = sel_data['detailed_label_num'].values

    # initialize the clustering algorithms
    hdbscan_boruvka = hdbscan.HDBSCAN(algorithm='boruvka_kdtree')
    hdbscan_boruvka.fit(x)
    print('Number of clusters identified: ' + str(hdbscan_boruvka.labels_.max()+1))
    print_results(hdbscan_boruvka.labels_, y, y_spec)

    # apply the elbow method for k-Means
    print('----------------------- Finding optimal number of clusters for k-Means -----------------------')
    Sum_of_squared_distances = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k)
        km = km.fit(x)
        Sum_of_squared_distances.append(km.inertia_)

    plt.figure()
    plt.plot(range(1, 11), Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.grid()
    plt.show()

    k = int(input('Enter your preferred number of clusters: '))
    k_means = KMeans(k).fit(x)
    print_results(k_means.labels_, y, y_spec)
