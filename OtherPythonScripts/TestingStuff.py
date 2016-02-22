

from os.path import join
import BrainDataAnalysis.Utilities as ut
import BrainDataAnalysis.ploting_functions as pf
import t_sne_bhcuda.t_sne_spikes as tsne_spikes
import t_sne_bhcuda.bhtsne_cuda as TSNE
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.spatial.distance as spdist
from sklearn.preprocessing import normalize
import h5py as h5
import numpy as np


base_directory = r'D:\Data\George\Projects\SpikeSorting\HarrisLab_SyntheticData'

data_set_dir = r'DataSet1'

kwx_file_path = join(base_directory, data_set_dir, 'testOutput.kwx')

spike_mat_file = join(base_directory, data_set_dir, r'20141202_all_es_gtTimes.mat')


# Get spike times
kwik_file_path = join(base_directory, data_set_dir, 'testOutput.kwik')
h5file = h5.File(kwik_file_path, mode='r')
spike_times = np.array(list(h5file['channel_groups/1/spikes/time_samples']))
h5file.close()

spike_times_half = spike_times[:spike_times.shape[0]/2]
spike_times_test = spike_times[spike_times.shape[0]/2:]

# Get clusters
mat_dict = sio.loadmat(spike_mat_file)
labeled_spike_times = mat_dict['gtTimes'][0]

spikes_labeled_dict = dict()
spikes_labeled_dict_half = dict()
spikes_labeled_dict_test = dict()
number_of_labels = labeled_spike_times.shape[0]

for i in range(number_of_labels):
    common_spikes, spikes_labeled_dict[i], labeled_spikes_not_found = \
        ut.find_points_in_array_with_jitter(labeled_spike_times[i][:, 0], spike_times, 6)
for i in range(number_of_labels):
    common_spikes_half, spikes_labeled_dict_half[i], labeled_spikes_not_found_half = \
        ut.find_points_in_array_with_jitter(labeled_spike_times[i][:, 0], spike_times_half, 6)
for i in range(number_of_labels):
    common_spikes_test, spikes_labeled_dict_test[i], labeled_spikes_not_found_test = \
        ut.find_points_in_array_with_jitter(labeled_spike_times[i][:, 0], spike_times_test, 6)



# Load t-sne results
tsne = np.load(join(base_directory, data_set_dir, 't_sne_result_p300_it5k_th02_eta200.npy'))
tsne_half = tsne[:, :tsne.shape[1]/2]


pf.plot_tsne(tsne, labels_dict=spikes_labeled_dict, label_name='Cell')
pf.plot_tsne(tsne_half, labels_dict=spikes_labeled_dict_half, subtitle='T-SNE Half points', label_name='Cell')


# Load the masked pca features
h5file = h5.File(kwx_file_path, mode='r')
pca_and_masks = np.array(list(h5file['channel_groups/1/features_masks']))
masks = np.array(pca_and_masks[:, :, 1])
pca_features = np.array(pca_and_masks[:, :, 0])
masked_pca_features = pca_features * masks
masked_pca_features_half = masked_pca_features[:masked_pca_features.shape[0]/2, :]



# Calculate distances between all pairs of features
euclidean_distances_between_half = spdist.pdist(masked_pca_features_half, 'euclidean')


test_point_index = masked_pca_features_half.shape[0]+4000
test_point = np.array([masked_pca_features[test_point_index, :]])
euclidean_distances_to_test_point = spdist.cdist(test_point, masked_pca_features_half, 'euclidean')

num_of_closest_points = 10
indices_of_closest_points = np.argsort(euclidean_distances_to_test_point)[:, :num_of_closest_points].squeeze()

distances_of_closest_points = euclidean_distances_to_test_point[:, indices_of_closest_points]
start = 0
end = 1
X_std = (distances_of_closest_points - distances_of_closest_points.min()) / \
        (distances_of_closest_points.max() - distances_of_closest_points.min())
weights_of_closest_points = np.squeeze(X_std * (start - end) + end)

center_of_mass = np.average(tsne[:, indices_of_closest_points], axis=1, weights=weights_of_closest_points)



fig, ax = pf.plot_tsne(tsne_half, labels_dict=spikes_labeled_dict_half, subtitle='T-SNE Half points', label_name='Cell')
ax.scatter(tsne[0][test_point_index], tsne[1][test_point_index], s=150, marker='o', alpha=0.5, c='r')
ax.scatter(tsne[0][indices_of_closest_points], tsne[1][indices_of_closest_points], s=300, marker='*', alpha=0.4, c='b')
ax.scatter(center_of_mass[0], center_of_mass[1], s=50, marker='p', alpha=0.4, c='g')




start = 0
end = 1

test_points = np.array([masked_pca_features[masked_pca_features.shape[0]/2:masked_pca_features.shape[0], :]]).squeeze()
euclidean_distances_to_test_points = spdist.cdist(test_points, masked_pca_features_half, 'euclidean')

num_of_closest_points = 10
indices_of_closest_points = np.argsort(euclidean_distances_to_test_points, axis=0)[:, :num_of_closest_points]
distances_of_closest_points = np.zeros(indices_of_closest_points.shape)
weights_of_closest_points = np.zeros(indices_of_closest_points.shape)
center_of_mass = np.zeros((2, indices_of_closest_points.shape[0]))
for i in range(0, indices_of_closest_points.shape[0]):
    distances_of_closest_points[i, :] = euclidean_distances_to_test_points[i, indices_of_closest_points[i, :]]
    X_std = (distances_of_closest_points[i, :] - distances_of_closest_points[i, :].min()) / \
        (distances_of_closest_points[i, :].max() - distances_of_closest_points[i, :].min())
    weights_of_closest_points[i, :] = np.squeeze(X_std * (start - end) + end)
    center_of_mass[:, i] = np.average(tsne_half[:, indices_of_closest_points[i, :]], axis=1,
                                      weights=weights_of_closest_points[i, :])



fig, ax = pf.plot_tsne(tsne_half, labels_dict=spikes_labeled_dict_half, subtitle='T-SNE Half points', label_name='Cell')
fig, ax = pf.plot_tsne(center_of_mass, labels_dict=spikes_labeled_dict_test, subtitle='T-SNE Half points', label_name='Cell')




#-----------------------------------------------------------------------------------------------------------------------
# Setting up test for t_sne_bhcuda extended
from os.path import join
import h5py as h5
import numpy as np
import scipy.io as sio
import BrainDataAnalysis.Utilities as ut
import t_sne_bhcuda.t_sne_spikes as tsne_spikes
import t_sne_bhcuda.bhtsne_cuda as TSNE
import matplotlib.pyplot as plt
import BrainDataAnalysis.ploting_functions as pf

base_directory = r'D:\Data\George\Projects\SpikeSorting\HarrisLab_SyntheticData'
data_set_num = 1
data_set_dir = r'DataSet{}'.format(data_set_num)
kwx_file_path = join(base_directory, data_set_dir, 'testOutput.kwx')
mat_file_dict = {1: '20141202_all_es_gtTimes.mat', 2: '20150924_1_e_gtTimes.mat',
                 3: '20150601_all_s_gtTimes.mat', 4: '20150924_1_GT_gtTimes.mat',
                 5: '20150601_all_GT_gtTimes.mat', 6: '20141202_all_GT_gtTimes.mat'}
spike_mat_file = join(base_directory, data_set_dir, mat_file_dict[data_set_num])

# Get spike times
kwik_file_path = join(base_directory, data_set_dir, 'testOutput.kwik')
h5file = h5.File(kwik_file_path, mode='r')
spike_times = np.array(list(h5file['channel_groups/1/spikes/time_samples']))
h5file.close()


# Get clusters
mat_dict = sio.loadmat(spike_mat_file)
labeled_spike_times = mat_dict['gtTimes'][0]

spikes_used = len(spike_times)

# 1) Get indices of labeled spikes
spikes_labeled_dict = dict()
number_of_labels = labeled_spike_times.__len__()
for i in range(number_of_labels):
    common_spikes, spikes_labeled_dict[i], labeled_spikes_not_found = \
        ut.find_points_in_array_with_jitter(labeled_spike_times[i][:, 0], spike_times[:spikes_used], 6)


# 2) Find how many spikes are labeled
number_of_labeled_spikes = 0
for i in range(number_of_labels):
    number_of_labeled_spikes += labeled_spike_times[i][:, 0].shape[0]


# Get the masked features to tsne
h5file = h5.File(kwx_file_path, mode='r')
pca_and_masks = np.array(list(h5file['channel_groups/1/features_masks']))
h5file.close()
masks = np.array(pca_and_masks[:, :, 1])
pca_features = np.array(pca_and_masks[:, :, 0])
masked_pca_features = pca_features * masks
# and save part of them to a data.dat file to be t-sned by the C++ code
path_debug = r'E:\George\SourceCode\Repos\t_sne_bhcuda\build\vs2013\t_sne_bhcuda'
path_release = r'E:\George\SourceCode\Repos\t_sne_bhcuda\bin\windows'
path = path_release
temp = masked_pca_features[:spikes_used, :]
seed = 40000
TSNE.save_data_for_tsne(temp, path, filename='data.dat', theta=0.5, perplexity=500, eta=200, no_dims=2,
                        iterations=1000, seed=seed, gpu_mem=0.8, verbose=2, randseed=-1)


tsne = tsne_spikes.t_sne_spikes(kwx_file_path, mask_data=True, path_to_save_tmp_data=None,
                                indices_of_spikes_to_tsne=range(10000),
                                use_scikit=False, perplexity=50.0, theta=0.5, iterations=1000, seed=0, gpu_mem=0.8,
                                no_dims=2, eta=200, early_exaggeration=4.0, randseed=-1, verbose=True)

# Load t-sne results
tsne = TSNE.load_tsne_result(path, 'result.dat')
tsne = np.transpose(tsne)
tsne = np.load(join(base_directory, data_set_dir, 't_sne_result_p800_it2k_th06_eta200.npy'))


# Run t-sne
perplexity = 600
theta = 0.2
iterations = 2000
gpu_mem = 0.8
eta = 200
seed_spikes = 10000
early_exaggeration = 4.0
tsne = tsne_spikes.t_sne_spikes(kwx_file_path, mask_data=True, perplexity=perplexity, theta=theta, iterations=iterations,
                                gpu_mem=gpu_mem, eta=eta, early_exaggeration=early_exaggeration)


fig = plt.figure()
ax = fig.add_subplot(111)
labeled_scatters = []
s = 10
ax.scatter(tsne[0][:seed], tsne[1][:seed], s=30, alpha=0.8)
ax.scatter(tsne[0][seed:], tsne[1][seed:], s=30, color='r', alpha=0.5)


pf.plot_tsne(tsne, labels_dict=spikes_labeled_dict, subtitle='T-SNE', label_name='Cell')