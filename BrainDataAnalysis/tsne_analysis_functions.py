

import numpy as np
import BrainDataAnalysis.ploting_functions as pf
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics



def split_juxta_spikes_into_groups_by_size(num_of_spike_groups, juxta_spike_peaks, juxta_spike_triggers):
    spike_thresholds_groups = np.arange(np.min(juxta_spike_peaks), np.max(juxta_spike_peaks),
                                        (np.max(juxta_spike_peaks) - np.min(juxta_spike_peaks)) / num_of_spike_groups)
    spike_thresholds_groups = np.append(spike_thresholds_groups, np.max(juxta_spike_peaks))
    juxta_spikes_grouped = {}
    juxta_spike_peaks_grouped = {}
    juxta_spike_triggers_grouped_withnans = {}
    juxta_spike_peaks_grouped_withnans = {}
    for t in range(1, len(spike_thresholds_groups)):
        juxta_spikes_grouped[t] = []
        juxta_spike_peaks_grouped[t] = []
        juxta_spike_peaks_grouped_withnans[t] = np.empty(len(juxta_spike_peaks))
        juxta_spike_peaks_grouped_withnans[t][:] = np.NAN
        juxta_spike_triggers_grouped_withnans[t] = np.empty(len(juxta_spike_peaks))
        juxta_spike_triggers_grouped_withnans[t][:] = np.NAN
    for s in range(len(juxta_spike_peaks)):
        for t in range(1, len(spike_thresholds_groups)):
            if juxta_spike_peaks[s] < spike_thresholds_groups[t]:
                juxta_spikes_grouped[t].append(juxta_spike_triggers[s])
                juxta_spike_peaks_grouped[t].append(juxta_spike_peaks[s])
                break
    for t in range(1, len(spike_thresholds_groups)):
        juxta_spike_peaks_grouped_withnans[t][np.in1d(juxta_spike_peaks, juxta_spikes_grouped[t])] = \
            juxta_spike_peaks[np.in1d(juxta_spike_peaks, juxta_spikes_grouped[t])]
        juxta_spike_triggers_grouped_withnans[t][np.in1d(juxta_spike_peaks, juxta_spikes_grouped[t])] = \
            juxta_spike_peaks[np.in1d(juxta_spike_peaks, juxta_spikes_grouped[t])]

    return juxta_spikes_grouped, juxta_spike_peaks_grouped, \
        juxta_spike_triggers_grouped_withnans, juxta_spike_peaks_grouped_withnans, spike_thresholds_groups


def fit_dbscan(data, eps, min_samples, show=True, juxta_cluster_indices_grouped=None, threshold_legend=None):
    X = np.transpose(data)

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    score = metrics.silhouette_score(X, labels, sample_size=5000)
    print('For eps={}, min_samples={}, estimated number of clusters={}'.format(eps, min_samples, n_clusters_))
    print("Silhouette Coefficient: {}".format(score))

    if show:
        pf.show_clustered_tsne(db, X, juxta_cluster_indices_grouped, threshold_legend)

    return db, n_clusters_, labels, core_samples_mask, score


def calculate_precision_recall_for_single_label_grouped(tsne, cluster_indices_grouped, n_clusters, labels,
                                                        core_samples_mask, show_means=False):
    # Define TP / FP / TN and FN
    X = np.transpose(tsne)
    juxta_cluster_indices = []
    num_of_spike_groups = len(cluster_indices_grouped)
    for g in range(0, num_of_spike_groups):
        juxta_cluster_indices.extend(cluster_indices_grouped[g])
    means_of_juxta = np.array([np.median(X[juxta_cluster_indices, 0]), np.median(X[juxta_cluster_indices, 1])])

    means_of_labels = np.zeros((n_clusters, 2))
    dmeans = np.zeros(n_clusters)
    for l in range(n_clusters):
        class_member_mask = (labels == l)
        xy = X[class_member_mask & core_samples_mask]
        means_of_labels[l, 0] = np.median(xy[:, 0])
        means_of_labels[l, 1] = np.median(xy[:, 1])
        dmeans[l] = np.linalg.norm((means_of_labels[l,:]-means_of_juxta))
    juxta_cluster_index = np.argmin(dmeans)

    # calculate prec, rec and f factor
    class_member_mask = (labels == juxta_cluster_index)
    extra_cluster_indices = [i for i, x in enumerate(class_member_mask & core_samples_mask) if x]
    tp_indices = np.intersect1d(juxta_cluster_indices, extra_cluster_indices)
    tp = len(tp_indices)
    all_pos = len(extra_cluster_indices)
    all_true = len(juxta_cluster_indices)
    precision = tp / all_pos
    recall = tp / all_true
    f_factor = 2*(precision*recall)/(precision+recall)
    print('Precision = {}, Recall = {}, F1 factor = {}'.format(precision, recall, f_factor))

    if show_means:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(means_of_juxta[0], means_of_juxta[1])
        ax.scatter(means_of_labels[:, 0], means_of_labels[:, 1], color='r')

    return precision, recall, f_factor


def calculate_precision_recal_for_many_labels(tsne, labels, core_samples_mask, n_clusters,
                                              db_labels, spikes_labeled_dict, show_means=False):
    #Define TP / FP / TN and FN
    X = np.transpose(tsne)
    number_of_labels = len(labels)
    means_of_labels = np.zeros((number_of_labels, 2))
    for l in range(number_of_labels):
        class_member_mask = (labels == l)
        xy = X[class_member_mask & core_samples_mask]
        means_of_labels[l, 0] = np.median(xy[:, 0])
        means_of_labels[l, 1] = np.median(xy[:, 1])

    means_of_DBclusters = np.zeros((n_clusters, 2))
    for c in range(n_clusters):
        class_member_mask = (db_labels == c)
        xy = X[class_member_mask & core_samples_mask]
        means_of_DBclusters[c, 0] = np.median(xy[:, 0])
        means_of_DBclusters[c, 1] = np.median(xy[:, 1])

    dmeans = np.zeros(n_clusters)
    labeled_cluster_db_index = []
    for l in range(number_of_labels):
        for c in range(n_clusters):
            dmeans[c] = np.linalg.norm((means_of_DBclusters[c, :]-means_of_labels[l]))
        labeled_cluster_db_index.append(np.argmin(dmeans))

    for l in range(number_of_labels):
        class_member_mask = (db_labels == labeled_cluster_db_index[l])
        db_cluster_indices = [i for i, x in enumerate(class_member_mask & core_samples_mask) if x]
        tp_indices = np.intersect1d(spikes_labeled_dict[l], db_cluster_indices)
        tp = len(tp_indices)
        all_pos = len(db_cluster_indices)
        all_true = len(spikes_labeled_dict[l])
        precision = tp / all_pos
        recall = tp / all_true
        f_factor = 2*(precision*recall)/(precision+recall)
        print('Precision = {}, Recall = {}, F1 factor = {}'.format(precision, recall, f_factor))

    if show_means:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(means_of_DBclusters[:, 0], means_of_DBclusters[:, 1])
        for i, txt in enumerate(range(n_clusters)):
            ax.annotate(str(txt), (means_of_DBclusters[i, 0], means_of_DBclusters[i, 1]), verticalalignment='top')
        ax.scatter(means_of_labels[:, 0], means_of_labels[:, 1], color='r')
        for i, txt in enumerate(range(number_of_labels)):
            ax.annotate(str(txt), (means_of_labels[i, 0], means_of_labels[i, 1]), verticalalignment='bottom')

    return precision, recall, f_factor

