

from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import sem, t
from sklearn.decomposition import pca
import pickle

from npeet.lnc import MI

from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data

import slider as sl

from BrainDataAnalysis.Statistics import binning
from BehaviorAnalysis import dlc_post_processing as dlc_pp

from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const

#  -------------------------------------------------
#  GET FOLDERS
#  -------------------------------------------------
date_folder = 8

dlc_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis', 'Deeplabcut')
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-05-07')

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')
spikes_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis', 'Denoised',
                     'Kilosort')
mutual_information_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis',
                                 'Results', 'MutualInformation')

num_of_frames_to_average = 0.25/(1/120)
#  -------------------------------------------------

#  -------------------------------------------------
#  GENERATE THE SPEED VECTOR
#  -------------------------------------------------

# Load the clean body markers
updated_markers_filename = join(dlc_project_folder, 'post_processing', 'cleaned_body_marker_positions_order_{}_gap_{}.df'.
                                format(str(4), str(10)))
updated_body_markers_positions = pd.read_pickle(updated_markers_filename)

# Use the markers to create a single body position
body_positions = dlc_pp.average_multiple_markers_to_single_one(updated_body_markers_positions, flip=True)
body_positions = dlc_pp.clean_large_movements_single_axis(body_positions, maximum_pixels=12)

# Use body position to create velocities (both linear and polar)
conversion_const = const.PIXEL_PER_FRAME_TO_METERS_PER_SECOND
body_velocities = np.diff(body_positions, axis=0) * conversion_const
body_velocities_polar = np.array([np.sqrt(np.power(body_velocities[:, 0], 2) + np.power(body_velocities[:, 1], 2)),
                         180 * (1/np.pi) * np.arctan2(body_velocities[:, 1], body_velocities[:, 0])]).transpose()

# Get the 250ms averages of firing rates for all neurons and of the speed (body_velocity_polar[0] of the animal
video_frame_spike_rates_filename = join(spikes_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates = np.load(video_frame_spike_rates_filename)
spike_rates_0p25 = np.load(join(spikes_folder, 'firing_rate_with_0p25s_window.npy'))

speeds_0p25 = binning.rolling_window_with_step(body_velocities_polar[:, 0], np.mean,
                                               num_of_frames_to_average, num_of_frames_to_average)

#  -------------------------------------------------
# CREATE THE MUTUAL INFORMATION MEASURE BETWEEN SPEED AND THE FIRING RATES OF ALL NEURONS (FOR THE WHOLE RECORDING)
#  -------------------------------------------------

# Calculate the mutual information between the speed and all firing rates (for the whole of the experiment)
# using the lnc code (a Kraskov with some local non-uniform correction for better very high correlations)
'''
n = 0
mutual_infos_spikes_vs_speed = []
for rate in spike_rates_0p25:
    mutual_infos_spikes_vs_speed.append(MI.mi_LNC([rate.tolist()[:const.BRAIN_DATA_UP_TO_QUARTER_SECOND], 
                                                                 speeds_0p25[:const.BRAIN_DATA_UP_TO_QUARTER_SECOND]],
                                        k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
    n += 1
    print('Done neuron {}'.format(str(n)))


mutual_infos_spikes_vs_speed = np.array(mutual_infos_spikes_vs_speed)
np.save(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed.npy'), mutual_infos_spikes_vs_speed)
'''


template_info = pd.read_pickle(join(spikes_folder, 'template_info.df'))
mutual_infos_spikes_vs_speed = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed.npy'))

speed_very_corr_neurons_index = np.squeeze(np.argwhere(mutual_infos_spikes_vs_speed > 0.15))
speed_very_corr_neurons = template_info.loc[speed_very_corr_neurons_index]

brain_positions_of_corr_neurons = speed_very_corr_neurons['position Y'].values * const.POSITION_MULT

plt.plot(np.array(spike_rates_0p25[speed_very_corr_neurons, :]).T)

# Calculate MI between the FRs of the two neurons
t1 = MI.mi_LNC([spike_rates_0p25[speed_very_corr_neurons[0], :].tolist(),
                spike_rates_0p25[speed_very_corr_neurons[1], :].tolist()],
               k=10, base=np.exp(1), alpha=0.4, intens=1e-10)

# Calculate MI between the FR of the same neuron and itself
t2 = MI.mi_LNC([spike_rates_0p25[speed_very_corr_neurons[1], :].tolist(),
                spike_rates_0p25[speed_very_corr_neurons[1], :].tolist()],
               k=10, base=np.exp(1), alpha=0.4, intens=1e-10)

# Calculate MI between the speed and itself
s = MI.mi_LNC([speeds_0p25, speeds_0p25],
               k=10, base=np.exp(1), alpha=0.4, intens=1e-10)

plt.plot(speeds_0p25)
plt.plot(spike_rates_0p25[speed_very_corr_neurons_index, :].T / 200)

plt.plot(binning.rolling_window_with_step(speeds_0p25, np.mean, 100, 1))
plt.plot(
    binning.rolling_window_with_step(spike_rates_0p25[speed_very_corr_neurons_index[0], :].T / 200, np.mean, 100, 1))

# Shuffle the spike rates of one of the best correlated neurons and calculate MI between the shuffled frs and speed
# (1000 times) to generate the basic chance base of MI
'''
speed_corr_neurons_index = [522]
shuffled, mean, conf_intervals = MI.shuffle_test(MI.mi_LNC,  spike_rates_0p25[speed_corr_neurons_index[0], :const.BRAIN_DATA_UP_TO_QUARTER_SECOND],
                                                 speeds_0p25[:const.BRAIN_DATA_UP_TO_QUARTER_SECOND], 
                                                 z=False, ns=1000, ci=0.95, k=10, base=np.exp(1), alpha=0.4, 
                                                 intens=1e-10)
np.save(join(mutual_information_folder, 'shuffled_mut_info_spike_rate_522_vs_speed.npy'), shuffled)
'''

# Have a look at the MIs vs the chance level
shuffled = np.load(join(mutual_information_folder, 'shuffled_mut_info_spike_rate_522_vs_speed.npy'))
mean_sh = np.mean(shuffled)
confidence_level = 0.99
confi_intervals = shuffled[int((1. - confidence_level) / 2 * 1000)], shuffled[int((1. + confidence_level) / 2 * 1000)]
# Log
plt.hist(mutual_infos_spikes_vs_speed, bins=np.logspace(np.log10(0.0001), np.log10(1), 50))
plt.hist(shuffled, bins=np.logspace(np.log10(0.0001), np.log10(1), 50), color=(1, 0, 0, 0.4))
plt.vlines([mean_sh, mean_sh+confi_intervals[0], mean_sh+confi_intervals[1]], 0, 60)
plt.gca().set_xscale("log")
# Linear
plt.hist(mutual_infos_spikes_vs_speed, bins= 200)
plt.hist(shuffled, bins=50, color=(1, 0, 0, 0.4))
plt.vlines([mean_sh, mean_sh+confi_intervals[0], mean_sh+confi_intervals[1]], 0, 60)


speed_corr_neurons_index = np.squeeze(np.argwhere(mutual_infos_spikes_vs_speed > mean_sh+confi_intervals[1]))
speed_corr_neurons = template_info.loc[speed_corr_neurons_index]

brain_positions_of_corr_neurons = np.array([speed_corr_neurons['position X'].values * const.POSITION_MULT,
                                            speed_corr_neurons['position Y'].values * const.POSITION_MULT])

plt.scatter(brain_positions_of_corr_neurons[0, :], brain_positions_of_corr_neurons[1, :],
            s=mutual_infos_spikes_vs_speed[speed_corr_neurons_index] * 200)


#  -------------------------------------------------
# USING IDTxl CHECK IF THERE IS A NETWORK OF INTERACTIONS BETWEEN ALL NEURONS AND SPEED
#  -------------------------------------------------
speed_very_corr_neurons_index = np.squeeze(np.argwhere(mutual_infos_spikes_vs_speed > 0.01))
second = [int(3.5 * 60 * 4), int(4.5 * 60 * 4)]  # Choose only one minute because it is too heavy to run on more data
full_array = np.vstack((speeds_0p25[second[0]:second[1]], spike_rates_0p25[speed_very_corr_neurons_index, second[0]:second[1]]))
data = Data(full_array, dim_order='ps')

network_analysis = MultivariateTE()
settings_cpu = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 8,
                'min_lag_sources': 1,
                'max_lag_target': 4
                }

settings_gpu = {'cmi_estimator': 'OpenCLKraskovCMI',
                'gpuid': 1,
                'max_lag_sources': 8,
                'min_lag_sources': 1,
                'max_lag_target': 4
                }

results = network_analysis.analyse_network(settings=settings_gpu, data=data)
# pickle.dump(results, open(join(mutual_information_folder, "one_second_of_network_interactions_3p4_tp_4p5_secs.pkl"), "wb" ) )
results = pickle.load( open(join(mutual_information_folder, "one_second_of_network_interactions_3p4_tp_4p5_secs.pkl"), "rb" ) )

#  -------------------------------------------------
# CREATE THE MUTUAL INFORMATION MEASURE BETWEEN POSITION AND THE FIRING RATES OF ALL NEURONS (FOR THE WHOLE RECORDING)
#  -------------------------------------------------

# Load the clean body markers
updated_markers_filename = join(dlc_project_folder, 'post_processing',
                                'cleaned_body_marker_positions_order_{}_gap_{}.df'.
                                format(str(4), str(10)))
updated_body_markers_positions = pd.read_pickle(updated_markers_filename)

# Use the markers to create a single body position
body_positions = dlc_pp.average_multiple_markers_to_single_one(updated_body_markers_positions, flip=True)
body_positions = dlc_pp.clean_large_movements_single_axis(body_positions, maximum_pixels=12)

# Discretize positions
body_positions_x_0p25 = np.array(binning.rolling_window_with_step(body_positions[:, 0], np.mean,
                                                                  num_of_frames_to_average, num_of_frames_to_average))
body_positions_y_0p25 = np.array(binning.rolling_window_with_step(body_positions[:, 1], np.mean,
                                                                  num_of_frames_to_average, num_of_frames_to_average))
body_positions_0p25 = np.array([[body_positions_x_0p25[i], body_positions_y_0p25[i]]
                                for i in np.arange(len(body_positions_x_0p25))])
# binned_body_positions_0p25 = binning.bin_2d_array(body_positions_0p25, bins=[10, 10]).get_values()
binned_body_positions_0p25 = binning.bin_2d_array(body_positions_0p25, bins=[100, 100]).get_values()


def prepare_data(d1_array, d2_array):
    if d2_array.shape[0] > d2_array.shape[1]:
        temp = np.vstack((np.expand_dims(d1_array, axis=1).transpose(),
                         d2_array.transpose()))
    else:
        temp = np.vstack((np.expand_dims(d1_array, axis=1),
                          d2_array))

    return temp.tolist()


p = pca.PCA()
pca_body_position = p.fit(body_positions_0p25.transpose()).components_[0]

# Calculate the mutual information between the position and all firing rates (for the whole of the experiment)
n = 0
mutual_infos_spikes_vs_position = []
for rate in spike_rates_0p25:
    mutual_infos_spikes_vs_position.append(MI.mi_LNC([rate.tolist()[:const.BRAIN_DATA_UP_TO_QUARTER_SECOND],
                                                      pca_body_position[:const.BRAIN_DATA_UP_TO_QUARTER_SECOND]],
                                                     k=10, base=np.exp(1), alpha=0.4, intens=1e-8))
    n += 1
    print('Done neuron {} with MI = {}'.format(str(n), str(mutual_infos_spikes_vs_position[-1])))


mutual_infos_spikes_vs_position = np.array(mutual_infos_spikes_vs_position)
np.save(join(mutual_information_folder, 'mutual_infos_spikes_vs_position.npy'), mutual_infos_spikes_vs_position)

mutual_infos_spikes_vs_position = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_position.npy'))

position_corr_neuron_index = 522
number_of_shuffles = 1000
shuffled, mean, conf_intervals = MI.shuffle_test(MI.mi_LNC,
                                                 spike_rates_0p25[position_corr_neuron_index,
                                                                  :const.BRAIN_DATA_UP_TO_QUARTER_SECOND],
                                                 pca_body_position[:const.BRAIN_DATA_UP_TO_QUARTER_SECOND],
                                                 z=False, ns=number_of_shuffles, ci=0.95, k=10, base=np.exp(1), alpha=0.4,
                                                 intens=1e-8)
np.save(join(mutual_information_folder, 'shuffled_mut_info_spike_rate_522_vs_position.npy'), shuffled)

shuffled = np.load(join(mutual_information_folder, 'shuffled_mut_info_spike_rate_522_vs_position.npy'))
mean_sh = np.mean(shuffled)
confidence_level = 0.95
confi_intervals = shuffled[int((1. - confidence_level) / 2 * number_of_shuffles)], \
                  shuffled[int((1. + confidence_level) / 2 * number_of_shuffles)]
# Log
plt.hist(mutual_infos_spikes_vs_position, bins=np.logspace(np.log10(0.0001), np.log10(1), 50))
plt.hist(shuffled, bins=np.logspace(np.log10(0.0001), np.log10(1), 50), color=(1, 0, 0, 0.4))
plt.vlines([mean_sh, mean_sh+confi_intervals[0], mean_sh+confi_intervals[1]], 0, 60)
plt.gca().set_xscale("log")
# Linear
plt.hist(mutual_infos_spikes_vs_position, bins= 200)
plt.hist(shuffled, bins=50, color=(1, 0, 0, 0.4))
plt.vlines([mean_sh, mean_sh+confi_intervals[0], mean_sh+confi_intervals[1]], 0, 60)


# Plot cells' firing rates on the arena



def conf_int(data):
    confidence = 0.99

    n = len(data)
    m = np.mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)

    start = m - h
    end = m + h
    return [start, end]



def plot_firing_rate_on_arena_tiles(neuron, figure):
    fr = spike_rates_0p25[neuron]
    image = np.zeros((10, 10))
    [start, end] = conf_int(fr)
    for x in np.arange(0, 10):
        for y in np.arange(0, 10):
            y_dot = y * 1000
            temp = fr[np.argwhere(binned_body_positions_0p25 == y_dot + x)]
            if len(temp) > 0:
                image[x, y] = np.mean(temp) / end

    figure.clear()
    ax = figure.add_subplot(111)
    ax.imshow(image, interpolation='bicubic', vmin=1, vmax=10)
    figure.suptitle('Avg. F.R. = {}\nTotal spikes = {}\nMax value = {}'.
                    format(str(np.mean(fr)), str(int(np.mean(fr) * len(fr) / 4)), str(image.max())))

neuron = 0
out = None
figure = plt.figure(0)
args = [figure]

sl.connect_repl_var(globals(), 'neuron', 'out', 'plot_firing_rate_on_arena_tiles', 'args',
                    slider_limits=[0, len(spike_rates_0p25) - 1])

#  -------------------------------------------------
# MI THE SKAGGS WAY
#  -------------------------------------------------
updated_markers_filename = join(dlc_project_folder, 'post_processing',
                                'cleaned_body_marker_positions_order_{}_gap_{}.df'.
                                format(str(4), str(10)))
updated_body_markers_positions = pd.read_pickle(updated_markers_filename)

# Use the markers to create a single body position
body_positions = dlc_pp.average_multiple_markers_to_single_one(updated_body_markers_positions, flip=True)
body_positions = dlc_pp.clean_large_movements_single_axis(body_positions, maximum_pixels=12)\
                    [:const.BRAIN_DATA_UP_TO_FRAME]
spike_rates = np.load(video_frame_spike_rates_filename)
spike_rates = spike_rates[:, :const.BRAIN_DATA_UP_TO_FRAME]


number_of_bins = 10
bins = np.arange(0, 640 + 640/number_of_bins, 640/number_of_bins)

occupancy = []
for i_x in np.arange(len(bins) - 1):
    x = [bins[i_x], bins[i_x+1]]
    for i_y in np.arange(len(bins) - 1):
        y = [bins[i_y], bins[i_y + 1]]
        occupancy.append(len(np.argwhere(np.logical_and(np.logical_and(body_positions[:, 0] > x[0],
                                                                       body_positions[:, 0] < x[1]),
                                                        np.logical_and(body_positions[:, 1] > y[0],
                                                                       body_positions[:, 1] < y[1])))))

occupancy = np.array(occupancy)/len(body_positions)


template_info = np.load()

spike_probabilities_at_position = []
for neuron in np.arange(len(spike_rates)):
    fr = spike_rates[neuron]
    mean_fr = np.mean(fr)
    fr_at_position = []
    for i_x in np.arange(len(bins) - 1):
        x = [bins[i_x], bins[i_x+1]]
        for i_y in np.arange(len(bins) - 1):
            y = [bins[i_y], bins[i_y + 1]]
            frames = np.argwhere(np.logical_and(np.logical_and(body_positions[:, 0] >= x[0],
                                                                           body_positions[:, 0] <= x[1]),
                                                            np.logical_and(body_positions[:, 1] > y[0],
                                                                           body_positions[:, 1] <= y[1])))
            if len(frames) > 0:
                fr_at_position.append(np.mean(fr[frames]))
            else:
                fr_at_position.append(0)
    spike_probabilities_at_position.append(np.array(fr_at_position) / mean_fr)

spike_probabilities_at_position = np.array(spike_probabilities_at_position)
np.save(join(mutual_information_folder, 'spike_probabilities_at_position.npy'), mutual_information_folder)

spike_probabilities_at_position = np.load(join(mutual_information_folder, 'spike_probabilities_at_position.npy'))


n = 0
mutual_infos_spikes_vs_position = []
for rate in spike_probabilities_at_position:
    mutual_infos_spikes_vs_position.append(MI.mi_LNC([rate.tolist(),
                                                      occupancy.tolist()],
                                                     k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
    n += 1
    print('Done neuron {} with MI = {}'.format(str(n), str(mutual_infos_spikes_vs_position[-1])))



mutual_infos_spikes_vs_position_skagg = []
for n in np.arange(len(spike_probabilities_at_position)):
    lambda_x = spike_probabilities_at_position[n]
    mean_fr = np.mean(spike_rates[n])
    mutual_infos_spikes_vs_position_skagg.append(np.nansum(lambda_x * mean_fr * np.log2(lambda_x) * occupancy))


shuffled_occupancy = np.copy(occupancy)
np.random.shuffle(shuffled_occupancy)

mutual_infos_spikes_vs_position_skagg_shuffled = []
for n in np.arange(len(spike_probabilities_at_position)):
    lambda_x = np.copy(spike_probabilities_at_position[n])
    np.random.shuffle(lambda_x)
    mean_fr = np.mean(spike_rates[n])
    mutual_infos_spikes_vs_position_skagg_shuffled.append(np.nansum(lambda_x * mean_fr * np.log2(lambda_x) * occupancy))

