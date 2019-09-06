

from os.path import join
import numpy as np
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p2 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
import BrainDataAnalysis.tsne_analysis_functions as tsne_funcs
from BrainDataAnalysis import binning
import matplotlib.pyplot as plt
from matplotlib import cm

import pandas as pd
from spikesorting_tsne import tsne, io_with_cpp as tsne_io

import sequence_viewer as sv
import transform as tr
import slider as sl

from sklearn.decomposition import PCA

import cv2

# -------------------------------------------------
# <editor-fold desc="LOAD FOLDERS">
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')

kilosort_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis',
                       'Kilosort')


analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')

tsne_folder_base = join(analysis_folder, 'Tsnes', 'Video')

subsumpled_video_folder = join(tsne_folder_base, 'SubsampledVideo')

barnes_hut_exe_dir = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\win\x64\Release'

events_folder = join(data_folder, "events")
event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)


spike_rates_per_video_frame_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates_per_video_frame = np.load(spike_rates_per_video_frame_filename)


dlc_folder = join(analysis_folder, 'Deeplabcut')
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-06-30')
video_file = join(dlc_folder, 'BonsaiCroping', 'Full_video.avi')

body_positions = np.load(join(dlc_project_folder, 'post_processing', 'body_positions.npy'))
speeds = np.load(join(dlc_project_folder, 'post_processing', 'speeds.npy'))

results_folder = join(analysis_folder, 'Results')
poke_folder = join(results_folder, 'EventsCorrelations', 'Poke')
events_definitions_folder = join(results_folder, 'EventsDefinitions')

template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

video_resolution = (100, 80)
capture = cv2.VideoCapture(video_file)
num_of_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="CREATE SUBSAMPLED VIDEO (RUN ONCE)">

grayscale_resized_video_array = []
for f in np.arange(num_of_frames):
    r, frame = capture.read()
    if not r:
        break
    f_mono = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f_mono = f_mono[64:576, :]
    f_mono_sub = cv2.resize(f_mono, (video_resolution[0], video_resolution[1]))
    grayscale_resized_video_array.append(f_mono_sub)

    if f%1000 == 0:
        print(f)

np.save(join(subsumpled_video_folder, 'grayscale_resized_video_array.npy'), grayscale_resized_video_array)


#    Have a look to see if resulting subsampled images are ok
grayscale_resized_video_array = np.load(join(subsumpled_video_folder, 'grayscale_resized_video_array.npy'))
c=grayscale_resized_video_array[0]
im=plt.imshow(c)
for pane in grayscale_resized_video_array:
    im.set_data(pane)
    plt.pause(0.02)

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="PCA THE 3D ARRAY (VIDEO) AND RUN T-SNE">

tsne_folder = join(tsne_folder_base, 'FullVideo_100ms')

grayscale_resized_video_array = np.load(join(subsumpled_video_folder, 'grayscale_resized_video_array.npy'))

#   Flatten
grayscale_resized_video_array = \
    grayscale_resized_video_array.reshape((num_of_frames-1, video_resolution[0] * video_resolution[1]))

#   Smooth over time to 100ms per frame
grayscale_resized_video_array_frame_smoothed = np.transpose(binning.rolling_window_with_step(grayscale_resized_video_array.transpose(),
                                                                                np.mean, 12, 12))
np.save(join(subsumpled_video_folder, 'grayscale_resized_video_array_frame_smoothed.npy'),
        grayscale_resized_video_array_frame_smoothed)

#    Have a look
frame = 0
t = grayscale_resized_video_array_frame_smoothed.reshape((44215, video_resolution[1], video_resolution[0]))
sv.image_sequence(globals(), 'frame', 'grayscale_resized_video_array_frame_smoothed')

#   PCA and keep the first 100 components
pca_flat_video = PCA(n_components=100)
pcs_flat_video = pca_flat_video.fit_transform(grayscale_resized_video_array_frame_smoothed)

#   Reverse PCA and have a look at the resolting images
rev_pcs_video = pca_flat_video.inverse_transform(pcs_flat_video)
rev_pcs_video = rev_pcs_video.reshape((rev_pcs_video.shape[0], video_resolution[1], video_resolution[0]))

plt.figure(1)
plt.imshow(rev_pcs_video[100, :, :])
plt.figure(2)
plt.imshow(grayscale_resized_video_array_frame_smoothed.reshape(rev_pcs_video.shape[0], video_resolution[1], video_resolution[0])[100,:,:])

#   Run the t-sne
num_dims = 2
perplexity = 100
theta = 0.3
eta = 200
exageration = 12
iterations = 4000
random_seed = 1
verbose = 3
tsne_result = tsne.t_sne(pcs_flat_video, tsne_folder, barnes_hut_exe_dir, num_dims=num_dims,
                                         perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                         iterations=iterations, random_seed=random_seed, verbose=verbose)

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="DBSCAN THE TSNE AND SAVE THE RESULTING LABELS">

tsne_result = tsne_io.load_tsne_result(tsne_folder)

X = tsne_result.transpose()

db, n_clusters_, labels, core_samples_mask, score = tsne_funcs.fit_dbscan(X, eps=0.025, min_samples=40,
                                                                          normalize=True, show=True)

np.save(join(tsne_folder, 'dbscan_labels.npy'), db.labels_)
# </editor-fold>
# -------------------------------------------------