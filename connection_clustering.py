import hdbscan
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


def select_connections(init_data, threshold=100):
    """
    Function for keeping only the flows with at least a threshold number of source-destination IP pairs in the data.
    Also some extra features are added, while the numerical representation of labels and detailed labels is introduced
    :param init_data: the initial data
    :param threshold: the threshold number of source-destination IP pairs
    :return: the selected data
    """
    connections_cnts = init_data.groupby(['src_ip', 'dst_ip']).agg(['count']).reset_index()
    sel_data = init_data.loc[(init_data['src_ip']
                              .isin(connections_cnts.loc[connections_cnts[('date', 'count')] > threshold]['src_ip'])) &
                             (init_data['dst_ip'].isin(connections_cnts.
                                                       loc[connections_cnts[('date', 'count')] > threshold]
                                                       ['dst_ip']))].reset_index(drop=True)
    sel_data['orig_packets_per_s'] = sel_data['orig_packets'] / sel_data['duration']
    sel_data['resp_packets_per_s'] = sel_data['resp_packets'] / sel_data['duration']
    sel_data['orig_bytes_per_s'] = sel_data['orig_ip_bytes'] / sel_data['duration']
    sel_data['resp_bytes_per_s'] = sel_data['resp_ip_bytes'] / sel_data['duration']
    sel_data['label_num'] = pd.Categorical(sel_data['label'], categories=sel_data['label'].unique()).codes
    sel_data['detailed_label_num'] = pd.Categorical(sel_data['detailed_label'],
                                                    categories=sel_data['detailed_label'].unique()).codes
    return sel_data


def outlier_removal(data, flag):
    """
    Function for removing outliers from the dataset either by checking the z-score (flag=1) or by checking the
    interquartile range (flag=2)
    :param data: the data to be examined
    :param flag: the method to be used for outlier removal
    :return: the data without the identified outliers
    """
    if flag == 1:
        # remove outliers using the z-score method
        z = np.abs(stats.zscore(data))
        data = data[(z < 3).all(axis=1)]
    else:
        # remove outliers using the IQR method
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        data = data[~((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).any(axis=1)]
    return data


def visualize_high_dimensional_data(data, labels):
    """
    Function for plotting high-dimensional by using the t-distributed Stochastic Neighbor Embedding to embed the data in
    two dimensions.
    :param data: the multidimensional dataset
    :param labels: the labels of the dataset
    :return:
    """
    projection = TSNE().fit_transform(data)
    color_palette = sns.color_palette('Paired', 12)
    cluster_colors = [color_palette[l] for l in labels]
    plt.figure()
    plt.scatter(projection[:, 0], projection[:, 1], s=50, linewidth=0, c=cluster_colors, alpha=0.25)
    plt.title('t-distributed Stochastic Neighbor Embedding of data')
    plt.show()


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

    print('------------------- Distribution of high level labels in clusters -------------------')
    for k, v in results_dict.items():
        print('the distribution of labels in cluster ' + str(k) + ' is:')
        for kk, vv in v.items():
            print(str(kk) + ': ' + str(vv))

    print('------------------- Distribution of specific labels in clusters -------------------')
    for k, v in result_spec_dict.items():
        print('the distribution of labels in cluster ' + str(k) + ' is:')
        for kk, vv in v.items():
            print(str(kk) + ': ' + str(vv))


if __name__ == '__main__':
    soomfy = pd.read_pickle('Datasets/IOT23/Benign-Soomfy-Doorlock/zeek_normal.pkl')
    amazon = pd.read_pickle('Datasets/IOT23/Benign-Amazon-Echo/zeek_normal.pkl')
    phillips = pd.read_pickle('Datasets/IOT23/Benign-Phillips-HUE/zeek_normal.pkl')

    # prepare benign data from each device for clustering
    sel_soomfy = select_connections(soomfy)
    soomfy_data = sel_soomfy[['src_port', 'protocol_num', 'orig_packets_per_s', 'resp_packets_per_s', 'duration']]
    sel_amazon = select_connections(amazon)
    amazon_data = sel_amazon[['src_port', 'protocol_num', 'orig_packets_per_s', 'resp_packets_per_s', 'duration']]
    sel_phillips = select_connections(phillips)
    phillips_data = sel_phillips[['src_port', 'protocol_num', 'orig_packets_per_s', 'resp_packets_per_s', 'duration']]

    # filepath_normal, filepath_malicious = input("Enter the desired filepaths (normal anomalous) separated by space: ").\
    #     split()
    filepath_normal = 'Datasets/IOT23/Malware-Capture-1-1/zeek_normal.pkl'
    filepath_malicious = 'Datasets/IOT23/Malware-Capture-1-1/zeek_anomalous.pkl'
    normal = pd.read_pickle(filepath_normal)
    anomalous = pd.read_pickle(filepath_malicious)

    # prepare mixed data for clustering
    mixed = pd.concat([normal, anomalous], ignore_index=True).sort_values(by='date').reset_index().drop(
        columns='index')
    sel_mixed = select_connections(mixed)
    mixed_data = sel_mixed[['src_port', 'protocol_num', 'orig_packets_per_s', 'resp_packets_per_s', 'duration']]

    rem = int(input('Select method for outlier removal (0: none | 1: z | 2: iqr): '))
    if rem:
        mixed_data = outlier_removal(mixed_data, rem)

    x = mixed_data.values
    # TODO: assign correct labels when the outliers are removed
    y = sel_mixed['label_num'].values
    y_spec = sel_mixed['detailed_label_num'].values

    # visualize_high_dimensional_data(x, y)

    # initialize the clustering algorithms
    hdbscan_boruvka = hdbscan.HDBSCAN(algorithm='boruvka_kdtree', min_cluster_size=200)
    hdbscan_boruvka.fit(x)
    print('Number of clusters identified: ' + str(hdbscan_boruvka.labels_.max() + 1))
    print_results(hdbscan_boruvka.labels_, y, y_spec)

    # apply the elbow method for k-Means
    print('----------------------- Finding optimal number of clusters for k-Means -----------------------')
    Sum_of_squared_distances = []
    for k in range(1, 15):
        km = KMeans(n_clusters=k)
        km = km.fit(x)
        Sum_of_squared_distances.append(km.inertia_)

    plt.figure()
    plt.plot(range(1, 15), Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.grid()
    plt.show()

    k = int(input('Enter your preferred number of clusters: '))
    k_means = KMeans(k).fit(x)
    print_results(k_means.labels_, y, y_spec)
