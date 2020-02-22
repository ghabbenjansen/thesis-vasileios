import hdbscan
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest


def select_connections(init_data, threshold=50):
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


def outlier_removal(data, method="z-score", feat=None, label=None):
    """
    Function for removing outliers from the dataset by checking the z-score (flag=1), the interquartile range (flag=2),
    or using the GLOSH outlier detection algorithm (flag=3)
    :param data: the data to be examined
    :param method: the method to be used for outlier removal
    :param feat: the features to be taken into account
    :param label: label of the dataset (needed for the GLOSH case)
    :return: the data without the identified outliers
    """
    # TODO: maybe add Amazon's Robust Random Cut Forest algorithm
    temp = data if feat is None else data[feat]

    if method == "z-score":
        # remove outliers using the z-score method
        z = np.abs(stats.zscore(temp))
        return data[(z < 3).all(axis=1)]
    elif method == "iqr":
        # remove outliers using the IQR method
        q1 = temp.quantile(0.25)
        q3 = temp.quantile(0.75)
        iqr = q3 - q1
        return data[~((temp < (q1 - 1.5 * iqr)) | (temp > (q3 + 1.5 * iqr))).any(axis=1)]
    elif method == "glosh":
        clusterer = hdbscan.HDBSCAN(min_cluster_size=20, metric='manhattan').fit(RobustScaler().fit_transform(temp))
        plt.figure()
        sns.distplot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=True)
        plt.title('Outlier score distribution for ' + label + ' data')
        plt.xlabel('Outlier score')
        plt.grid()
        plt.show()
        threshold = pd.Series(clusterer.outlier_scores_).quantile(0.9)
        return data.iloc[np.where(clusterer.outlier_scores_ <= threshold)[0]]
    elif method == "isolation-forest":
        clf = IsolationForest(random_state=1)
        preds = clf.fit_predict(temp)
        return data[(preds+1).astype(bool)]
    elif method == "rolling_median":
        roll_medians = pd.DataFrame()
        for col in temp.columns.values.tolist():
            roll_medians[col] = temp[col].rolling(3).median().fillna(method='bfill').fillna(method='ffill')
        normal_idx = np.abs(temp - roll_medians) < 3
        return data[np.rint(normal_idx.mean(axis=1)).astype(bool)]
    else:
        return data


def cluster_data(x, method='kmeans', k=2, c=100, function_kwds=None):
    """
    Function that fits the desired clustering method on the training data
    :param x: the training data
    :param method: the type of method (kmeans | hdbscan)
    :param k: the number of clusters if kmeans is selected
    :param c: the minimum cluster size if hdbscan is selected
    :param function_kwds: some function keywords (only for hdbscan currently)
    :return: the fitted clustering model
    """
    if function_kwds is None:
        function_kwds = {}
    return KMeans(k).fit(x) if method == 'kmeans' else hdbscan.HDBSCAN(algorithm='boruvka_kdtree', min_cluster_size=c,
                                                                       metric='manhattan', cluster_selection_epsilon=0.75,
                                                                       **function_kwds).fit(x)


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
    """
    Function for printing the distribution of the real sample labels in the predicted clusters
    :param predicted: the predicted cluster labels of each sample
    :param real: the real labels of each sample (0 for benign, 1 for malicious)
    :param real_spec: the specific labels of each sample (TODO:a more informational mapping to be added)
    :return: prints the results
    """
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


def select_clusters(predictions, criterion='top'):
    """
    Function for selecting the flows to comprise the bening training set of the system from the clustered ones
    :param predictions: a numpy array with the cluster labels of each flow
    :param criterion: the criterion to be used to select the clusters (TODO: add more robust criteria)
    :return: the boolean mask of the selected flows
    """
    # create a dict with the number of flows in each cluster - exclude the outliers/noise cluster of -1
    cnt_dict = dict([(i, list(predictions).count(i)) for i in range(np.max(predictions)+1)])
    # retrieve the total number of flows in the clusters
    tot_flows = sum([v for v in cnt_dict.values()])
    # the clusters' indices to be kept
    clusters_indices = []
    # if criterion equals to 'top' then keep the flows belonging to the most popular clusters and accounting for the
    # top 30% of the total flows clustered
    if criterion == 'top':
        s = 0
        for k, v in sorted(cnt_dict.items(), key=lambda item: item[1], reverse=True):
            if s <= int(0.3*tot_flows):
                clusters_indices += [k]
                s += v
    else:
        clusters_indices = [max(cnt_dict, key=cnt_dict.get)]
    return np.isin(predictions, clusters_indices)


if __name__ == '__main__':
    # First choose the way of clustering
    ans = int(input('Choose how to cluster the data\n'
                '0: create clusters based on the benign IOT datasets and fit these clusters on the mixed datasets'
                '\n1: cluster the mixed datasets directly\nGive an answer: '))

    # initialize the parameters to use
    selected = ['src_port', 'dst_port', 'protocol_num', 'orig_ip_bytes', 'resp_ip_bytes', 'duration']
    outlier_dict = {0: 'none', 1: 'z-score', 2: 'iqr', 3: 'glosh', 4: 'isolation-forest', 5: 'rolling-median'}

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

    if not ans:
        soomfy = pd.read_pickle('Datasets/IOT23/Benign-Soomfy-Doorlock/zeek_normal.pkl')
        amazon = pd.read_pickle('Datasets/IOT23/Benign-Amazon-Echo/zeek_normal.pkl')
        phillips = pd.read_pickle('Datasets/IOT23/Benign-Phillips-HUE/zeek_normal.pkl')

        # prepare benign data from each device for clustering
        sel_soomfy = select_connections(soomfy, 10)
        sel_amazon = select_connections(amazon)
        sel_phillips = select_connections(phillips, 10)

        # apply outlier removal to the benign flows
        removal_method = outlier_dict[int(input(
                'Select method for outlier removal (0: none | 1: z-score | 2: iqr | 3: GLOSH | 4: isolation-forest | '
                '5: rolling-median): '))]
        data_name = 'Soomfy doorlock' if removal_method == "glosh" else None
        soomfy_data = outlier_removal(sel_soomfy, removal_method, ['duration', 'orig_packets', 'orig_ip_bytes',
                                                                   'orig_packets_per_s', 'orig_bytes_per_s'], data_name)
        data_name = 'Amazon echo' if removal_method == "glosh" else None
        amazon_data = outlier_removal(sel_amazon, removal_method, ['duration', 'orig_packets', 'orig_ip_bytes',
                                                                   'resp_packets', 'resp_ip_bytes', 'protocol_num',
                                                                   'state_num', 'orig_packets_per_s',
                                                                   'resp_packets_per_s', 'orig_bytes_per_s',
                                                                   'resp_bytes_per_s'], data_name)
        data_name = 'Phillips hue' if removal_method == "glosh" else None
        phillips_data = outlier_removal(sel_phillips, removal_method, ['duration', 'orig_ip_bytes', 'resp_ip_bytes',
                                                                       'orig_packets_per_s', 'resp_packets_per_s',
                                                                       'orig_bytes_per_s', 'resp_bytes_per_s'],
                                        data_name)

        # initialize the clustering algorithms for the benign samples
        soomfy_clusters = cluster_data(soomfy_data[selected].values, method='hdbscan', c=30,
                                       function_kwds={'allow_single_cluster': True, 'prediction_data': True})
        print('Number of clusters identified for soomfy: ' + str(soomfy_clusters.labels_.max() + 1) + ' with ' +
              str(sum(soomfy_clusters.labels_ == 0)))
        amazon_clusters = cluster_data(amazon_data[selected].values, method='hdbscan', c=100,
                                       function_kwds={'allow_single_cluster': True, 'prediction_data': True})
        print('Number of clusters identified for amazon: ' + str(amazon_clusters.labels_.max() + 1) + ' with ' +
              str(sum(amazon_clusters.labels_ == 0)))
        phillips_clusters = cluster_data(phillips_data[selected].values, method='hdbscan', c=20,
                                         function_kwds={'allow_single_cluster': True, 'prediction_data': True})
        print('Number of clusters identified for phillips: ' + str(phillips_clusters.labels_.max() + 1) + ' with ' +
              str(sum(phillips_clusters.labels_ == 0)))

        mixed_data = sel_mixed[selected]
        x = mixed_data.values
        y = sel_mixed['label_num'].values
        y_spec = sel_mixed['detailed_label_num'].values

        # Visualize the dataset using TSNE
        # visualize_high_dimensional_data(x, y)

        # find clusters based on the fitted models for the benign devices
        for cl, iot in zip([soomfy_clusters, amazon_clusters, phillips_clusters], ['soomfy', 'amazon', 'phillips']):
            print(
                '--------------------------- Trying to fit ' + iot + ' clusters on the data ---------------------------')
            test_labels, strengths = hdbscan.approximate_predict(cl, x)
            print_results(test_labels, y, y_spec)

        # TODO: check how to find clusters in this case
    else:
        # first apply outlier removal
        removal_method = outlier_dict[
            int(input(
                'Select method for outlier removal (0: none | 1: z-score | 2: iqr | 3: GLOSH | 4: isolation-forest | '
                '5: rolling-median): '))]
        data_name = filepath_normal.split('/')[2] if removal_method == "glosh" else None
        mixed_data = outlier_removal(sel_mixed, removal_method, selected, data_name)

        # then cluster the remaining data
        mixed_clusters = cluster_data(mixed_data[selected].values, method='hdbscan', c=150)
        y = mixed_data['label_num'].values
        y_spec = mixed_data['detailed_label_num'].values

        # and print the results of this clustering
        print('Number of identified clusters: ' + str(max(mixed_clusters.labels_)+1))
        print_results(mixed_clusters.labels_, y, y_spec)

        # finally select the flows to be later used for training
        fin_selected_data = mixed_data[select_clusters(mixed_clusters.labels_, criterion='top')]

        # check the impurity value of the selected flows
        print(fin_selected_data['label_num'].value_counts())

        # and save the selected dataframe of flows to pickle for further use
        fin_selected_data.to_pickle('/'.join(filepath_normal.split('/')[0:2] + ['training', 'train_set.pkl']))


    # apply the elbow method for k-Means
    # print('----------------------- Finding optimal number of clusters for k-Means -----------------------')
    # Sum_of_squared_distances = []
    # for k in range(1, 15):
    #     km = KMeans(n_clusters=k)
    #     km = km.fit(x)
    #     Sum_of_squared_distances.append(km.inertia_)
    #
    # plt.figure()
    # plt.plot(range(1, 15), Sum_of_squared_distances, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Sum_of_squared_distances')
    # plt.title('Elbow Method For Optimal k')
    # plt.grid()
    # plt.show()
    #
    # k = int(input('Enter your preferred number of clusters: '))
    # k_means = KMeans(k).fit(x)
    # print_results(k_means.labels_, y, y_spec)
