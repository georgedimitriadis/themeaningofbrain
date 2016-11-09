__author__ = 'George Dimitriadis'

from os.path import join
from os import rename
import t_sne_bhcuda.t_sne_spikes as tsne_spikes
import t_sne_bhcuda.bhtsne_cuda as TSNE
import BrainDataAnalysis.Utilities as ut
import BrainDataAnalysis.ploting_functions as pf
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import scipy.io as sio
import h5py as h5
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics

base_directory = r'D:\Data\George\Projects\SpikeSorting\HarrisLab_SyntheticData'

data_set_num = 5
data_set_dir = r'DataSet{}'.format(data_set_num)

results_dir = join(base_directory, data_set_dir, 'tsne_results')
kwx_file_path = join(base_directory, data_set_dir, 'testOutput.kwx')

mat_file_dict = {1: '20141202_all_es_gtTimes.mat', 2: '20150924_1_e_gtTimes.mat',
                 3: '20150601_all_s_gtTimes.mat', 4: '20150924_1_GT_gtTimes.mat',
                 5: '20150601_all_GT_gtTimes.mat', 6: '20141202_all_GT_gtTimes.mat'}
spike_mat_file = join(base_directory, data_set_dir, mat_file_dict[data_set_num])

tsne_video_path = join(base_directory, data_set_dir, 'video_allspikes_seed0')

# Get spike times
kwik_file_path = join(base_directory, data_set_dir, 'testOutput.kwik')
h5file = h5.File(kwik_file_path, mode='r')
spike_times = np.array(list(h5file['channel_groups/1/spikes/time_samples']))
h5file.close()

spikes_used = 400000#len(spike_times)#130000

# Get clusters
mat_dict = sio.loadmat(spike_mat_file)
labeled_spike_times = mat_dict['gtTimes'][0]

# 1) Get indices of labeled spikes
spikes_labeled_dict = dict()
number_of_labels = labeled_spike_times.__len__()
for i in range(number_of_labels):
    common_spikes, spikes_labeled_dict[i], labeled_spikes_not_found = \
        ut.find_points_in_array_with_jitter(labeled_spike_times[i][:, 0], spike_times[:spikes_used], 6)



# 2) Generate a labels array (each spike is represented by its label number or -1 if it doesn't have a label
labels = np.zeros(spikes_used)
labels[0:] = -1
for l in range(number_of_labels):
    labels[spikes_labeled_dict[l]] = l


# 3) Find how many spikes are labeled
number_of_labeled_spikes = 0
for i in range(number_of_labels):
    number_of_labeled_spikes += labeled_spike_times[i][:, 0].shape[0]



# Run t-sne
path_to_save_tmp_data = tsne_video_path
perplexity = 200
theta = 0.2
iterations = 5000
gpu_mem = 0.9
eta = 200
early_exaggeration = 4.0
seed = 400000
verbose = 3
randseed = 0
tsne = tsne_spikes.t_sne_spikes(kwx_file_path, path_to_save_tmp_data=path_to_save_tmp_data,
                                hdf5_dir_to_pca=r'channel_groups/1/features_masks',
                                mask_data=True, perplexity=perplexity, theta=theta, iterations=iterations,
                                gpu_mem=gpu_mem, seed=seed, eta=eta, early_exaggeration=early_exaggeration,
                                verbose=verbose, indices_of_spikes_to_tsne=range(spikes_used), randseed=randseed)



# Load t-sne results
tsne = TSNE.load_tsne_result(results_dir, 'result_tsne40K_com46k_p500_it1k_th05_eta200.dat')
tsne = np.transpose(tsne)
tsne = np.load(join(results_dir, 't_sne_results_s130k_100per_200lr_02theta.npy'))


# 2D plot
pf.plot_tsne(tsne, labels_dict=spikes_labeled_dict, subtitle='T-sne of first 130k spikes from Synthetic Data',
             label_name='"Cell" No', cm=plt.cm.jet, markers=['.', '^'], sizes=[3, 20])
pf.plot_tsne(tsne, labels_dict=None, subtitle='T-sne of 86000 spikes from Synthetic Data, not labeled', label_name=None)



#--------------------------------------------------------------------------------------
# CHECK QUALITY OF FIT
# 1) DBSCAN
def fit_dbscan(data, eps, min_samples, show=True, juxta_cluster_indices_grouped=None, threshold_legend=None):
    X = np.transpose(data)

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    db_labels = db.labels_

    # Number of clusters in db_labels, ignoring noise if present.
    n_clusters_ = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    score = metrics.silhouette_score(X, db_labels, sample_size=5000)
    print('For eps={}, min_samples={}, estimated number of clusters={}'.format(eps, min_samples, n_clusters_))
    print("Silhouette Coefficient: {}".format(score))

    if show:
        pf.show_clustered_tsne(db, X, juxta_cluster_indices_grouped, threshold_legend)

    return db, n_clusters_, db_labels, core_samples_mask, score


db, n_clusters_, db_labels, core_samples_mask, score = fit_dbscan(tsne, 0.2, 4, show=True)


# 2) Define TP / FP / TN and FN
X = np.transpose(tsne)

means_of_labels = np.zeros((number_of_labels, 2))
for l in range(number_of_labels):
    class_member_mask = (labels == l)
    xy = X[class_member_mask & core_samples_mask]
    means_of_labels[l, 0] = np.median(xy[:, 0])
    means_of_labels[l, 1] = np.median(xy[:, 1])

means_of_DBclusters = np.zeros((n_clusters_, 2))
for c in range(n_clusters_):
    class_member_mask = (db_labels == c)
    xy = X[class_member_mask & core_samples_mask]
    means_of_DBclusters[c, 0] = np.median(xy[:, 0])
    means_of_DBclusters[c, 1] = np.median(xy[:, 1])


dmeans = np.zeros(n_clusters_)
labeled_cluster_db_index = []
for l in range(number_of_labels):
    for c in range(n_clusters_):
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


# 2.5) have a look where the averages are
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(means_of_DBclusters[:, 0], means_of_DBclusters[:, 1])
for i, txt in enumerate(range(n_clusters_)):
    ax.annotate(str(txt), (means_of_DBclusters[i, 0], means_of_DBclusters[i, 1]), verticalalignment='top')
ax.scatter(means_of_labels[:, 0], means_of_labels[:, 1], color='r')
for i, txt in enumerate(range(number_of_labels)):
    ax.annotate(str(txt), (means_of_labels[i, 0], means_of_labels[i, 1]), verticalalignment='bottom')
#--------------------------------------------------------------------------------------



#---------------------------------------------------------------------------
# MAKE VIDEO OF TSNE

video_dir = r'D:\Data\George\Projects\SpikeSorting\HarrisLab_SyntheticData\DataSet5\video_allspikes_seed0'
iters = np.arange(5000)
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='T-sne movie', artist='George Dimitriadis')
writer = FFMpegWriter(fps=30, bitrate=-1, metadata=metadata)
number_of_labels = spikes_labeled_dict.__len__()
color_indices = plt.Normalize(0, number_of_labels)
cm = plt.cm.Dark2
fig = plt.figure(figsize=(15, 15), dpi=200)
ax = fig.add_subplot(111)
with writer.saving(fig, join(video_dir, 'writer_test.mp4'), 200):
    for it in iters:
        ax.cla()
        tsne = TSNE.load_tsne_result(video_dir, 'interim_{:0>6}.dat'.format(it))
        tsne = np.transpose(tsne)
        pf.plot_tsne(tsne, axes=ax, labels_dict=spikes_labeled_dict, cm=plt.cm.jet, markers=['.', '^'], sizes=[1, 6], max_screen=False)
        minX = np.min(tsne[0, :])
        maxX = np.max(tsne[0, :])
        minY = np.min(tsne[1, :])
        maxY = np.max(tsne[1, :])
        rangeX = np.max(np.abs([minX, maxX]))
        rangeY = np.max(np.abs([minY, maxY]))
        # print(rangeX)
        plt.ylim([-rangeY, rangeY])
        plt.xlim([-rangeX, rangeX])
        writer.grab_frame()


#OR
video_dir = r'D:\Data\George\Projects\SpikeSorting\HarrisLab_SyntheticData\DataSet5\video_allspikes_seed0'
iterations = 5000
cm = plt.cm.jet
markers = ['.', '^']
sizes = [1, 6]
pf.make_video_of_tsne_iterations(iterations, video_dir, data_file_name='interim_{:0>6}.dat',
                                 video_file_name='tsne_video.mp4', figsize=(15, 15), dpi=200, fps=30,
                                 labels_dict=spikes_labeled_dict, cm=cm, sizes=sizes, markers=markers)
#------------------------------------------------------------------------
