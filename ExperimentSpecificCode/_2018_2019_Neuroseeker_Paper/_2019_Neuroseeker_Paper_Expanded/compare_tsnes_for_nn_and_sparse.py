

from os.path import join
import numpy as np
import pandas as pd
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_25_AK_47p2 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
import matplotlib.pyplot as plt
from matplotlib import cm

import pandas as pd
from spikesorting_tsne import tsne, io_with_cpp as tsne_io
from scipy.spatial.distance import cdist


import matplotlib
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

# -------------------------------------------------
# <editor-fold desc="LOAD FOLDERS">

rats = {1:'33.1', 2:'47.2', 3:'47.1'}

tsne_folders_nn = {'33.1': r'Y:\swc\kampff\George\DataAndResults\Experiments\Awake\NeuroSeeker\AK_33.1' +
                             r'\2018_04_30-11_38\Analysis\Tsne\SpikingVectors\All_spikes_100msbin_count_top40PCs_6Kiters',
                   '47.2': r'Y:\swc\kampff\George\DataAndResults\Experiments\Awake\NeuroSeeker\AK_47.2\2019_06_25-12_50'+
                             r'\Analysis\Tsnes\SpikingVectors\All_spikes_100msbin_count_top40PCs_10Kiter',
                   '47.1': r'Y:\swc\kampff\George\DataAndResults\Experiments\Awake\NeuroSeeker\AK_47.1'+
                             r'\2019_07_04-11_51\Analysis\Tsnes\SpikingVectors\All_spikes_100msbin_count_top40PCs_6Kiter'
                   }

tsne_folders_sp = {'33.1': r'Y:\swc\kampff\George\DataAndResults\Experiments\Awake\NeuroSeeker\AK_33.1\2018_04_30-11_38'+
                             r'\Analysis\NeuropixelSimulations\Sparce\Results\Tsnes\SpikingVectors'+
                             r'\All_spikes_100msbin_count_top40PCs_6Kiter',
                   '47.2': r'Y:\swc\kampff\George\DataAndResults\Experiments\Awake\NeuroSeeker\AK_47.2\2019_06_25-12_50'+
                            r'\Analysis\NeuropixelSimulations\Sparce\Results\Tsnes\SpikingVectors'+
                            r'\All_spikes_100msbin_count_top40PCs_10Kiter',
                   '47.1': r'Y:\swc\kampff\George\DataAndResults\Experiments\Awake\NeuroSeeker\AK_47.1\2019_07_04-11_51'+
                            r'\Analysis\NeuropixelSimulations\Sparce\Results\Tsnes\SpikingVectors'+
                            r'\All_spikes_100msbin_count_top40PCs_6Kiter'
                  }

video_tsne_folders = {'33.1': r'Y:\swc\kampff\George\DataAndResults\Experiments\Awake\NeuroSeeker\AK_33.1'+
                              r'\2018_04_30-11_38\Analysis\Tsne\Video\CroppedVideo_100ms_Top100PCs',
                      '47.2': r'Y:\swc\kampff\George\DataAndResults\Experiments\Awake\NeuroSeeker\AK_47.2'+
                              r'\2019_06_25-12_50\Analysis\Tsnes\Video\CroppedVideo_100ms_Top100PCs',
                      '47.1': r'Y:\swc\kampff\George\DataAndResults\Experiments\Awake\NeuroSeeker\AK_47.1'+
                              r'\2019_07_04-11_51\Analysis\Tsnes\Video\CroppedVideo_100ms_Top100PCs'}

tsne_results_nn = {}
tsne_results_sp = {}
video_labels = {}
for r in rats:
    rat = rats[r]
    tsne_results_nn[rat] = tsne_io.load_tsne_result(tsne_folders_nn[rat])
    tsne_results_sp[rat] = tsne_io.load_tsne_result(tsne_folders_sp[rat])
    video_labels[rat] = np.load(join(video_tsne_folders[rat], 'dbscan_labels.npy'))


mean_distances_of_labels_nn = {}
mean_distances_of_labels_sp = {}

for r in rats:
    rat = rats[r]
    unique_labels = np.unique(video_labels[rat])
    unique_labels = np.delete(unique_labels, np.argwhere(unique_labels==-1))

    nn_tsne = tsne_results_nn[rat]
    random_selection_of_tsne_points_indices = np.random.choice(np.arange(len(nn_tsne)), 20000)
    random_selection_of_tsne_points = nn_tsne[random_selection_of_tsne_points_indices, :]
    mean_distance_of_all = np.mean(cdist(random_selection_of_tsne_points, random_selection_of_tsne_points))

    mean_distances_of_labels = []
    for l in unique_labels:
        tsne_of_label = nn_tsne[video_labels[rat] == l]
        mean_distances_of_labels.append(np.mean(cdist(tsne_of_label, tsne_of_label)) / mean_distance_of_all)

    mean_distances_of_labels_nn[rat] = mean_distances_of_labels

    print('Done NN of rat {}'.format(rat))

    sp_tsne = tsne_results_sp[rat]
    random_selection_of_tsne_points_indices = np.random.choice(np.arange(len(sp_tsne)), 20000)
    random_selection_of_tsne_points = sp_tsne[random_selection_of_tsne_points_indices, :]
    mean_distance_of_all = np.mean(cdist(random_selection_of_tsne_points, random_selection_of_tsne_points))

    mean_distances_of_labels = []
    for l in unique_labels:
        tsne_of_label = sp_tsne[video_labels[rat] == l]
        mean_distances_of_labels.append(np.mean(cdist(tsne_of_label, tsne_of_label)) / mean_distance_of_all)

    mean_distances_of_labels_sp[rat] = mean_distances_of_labels

    print('Done SP of rat {}'.format(rat))


mean_distances_of_labels = []
percentage_under_half_nn = []
percentage_under_half_sp = []
under_half_nn = []
under_half_sp = []
for r in rats:
    rat = rats[r]
    mean_distances_of_labels.append(mean_distances_of_labels_nn[rat])
    under_half_nn.append(len(np.where(np.array(mean_distances_of_labels_nn[rat]) < 0.54)[0]))
    percentage_under_half_nn.append(len(np.where(np.array(mean_distances_of_labels_nn[rat]) < 0.5)[0]) /
                                    len(np.array(mean_distances_of_labels_nn[rat])))

    mean_distances_of_labels.append(mean_distances_of_labels_sp[rat])
    under_half_sp.append(len(np.where(np.array(mean_distances_of_labels_sp[rat]) < 0.54)[0]))
    percentage_under_half_sp.append(len(np.where(np.array(mean_distances_of_labels_sp[rat]) < 0.5)[0]) /
                                    len(np.array(mean_distances_of_labels_sp[rat])))


_ = plt.boxplot(mean_distances_of_labels, notch=True, bootstrap=10000)
plt.xticks(np.arange(1, 7), ['Animal 1 \nNeuroSeeker', 'Animal 1 \nSparse 4 shanks', 'Animal 2 \nNeuroSeeker',
                             'Animal 2 \nSparse 4 shanks', 'Animal 3 \nNeuroSeeker', 'Animal 3 \nSparse 4 shanks'],
           fontname='Arial')
plt.ylabel('Average distance of cluster points / \ndistance of full embedding', size=24, fontname='Arial')
