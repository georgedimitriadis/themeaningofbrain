
import numpy as np
import matplotlib.pyplot as plt
import Layouts.Probes.probes_imec as pr_im
import BrainDataAnalysis.Graphics.ploting_functions as pf
import mne.filter as filters
import scipy.ndimage.measurements as sp_m





window = 0
window_size_secs = 10
filtered_data_type = np.float64
sampling_freq = 30000
high_pass_freq = 500
window_size = int(window_size_secs * sampling_freq)
iir_params = {'order': 4, 'ftype': 'butter', 'padlen': 0}
num_of_channels = np.shape(raw_data_ivm.dataMatrix)[0]
electrode_structure, channel_positions = pr_im.create_128channels_imec_prb()


# Get the high passed data for the current time window
temp_unfiltered = raw_data_ivm.dataMatrix[:, window * window_size:(window + 1) * window_size]
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_filtered = filters.high_pass_filter(temp_unfiltered,
                                         sampling_freq, high_pass_freq, method='iir',
                                             iir_params=iir_params)


# Find thresholds
stdvs  = np.median(np.abs(temp_filtered)/0.6745, axis=1)

large_thresholds = np.zeros(np.shape(temp_filtered))
small_thresholds = np.zeros(np.shape(temp_filtered))
for c in range(num_of_channels):
    large_thresholds[c, :] = 7 * stdvs[c]
    small_thresholds[c, :] = 2 * stdvs[c]

# Generate thresholded array of -1 (if negative threshold is passed), +1 (if possitive threshold is passed)
# and 0 otherwise
threshold_crossing_regions = np.zeros(np.shape(temp_filtered))
threshold_crossing_regions[temp_filtered < -large_thresholds] = -1
threshold_crossing_regions[temp_filtered > large_thresholds] = 1



# Put the thresholded data on the 2D probe with time the 3rd dimension
on_probe_shape = tuple(np.concatenate((np.shape(electrode_structure), [np.shape(threshold_crossing_regions)[1]])))
threshold_crossing_regions_on_probe = np.zeros(on_probe_shape)
for r in range(32):
    for c in range(4):
        threshold_crossing_regions_on_probe[r, c, :] = threshold_crossing_regions[electrode_structure[r, c], :]

labels, num_of_features = sp_m.label(threshold_crossing_regions_on_probe)
object_slices = sp_m.find_objects(labels)

channels_in_each_label = []
for i in range(num_of_features):
    start_channel = object_slices[i][0].start * 4 + object_slices[i][1].start
    end_channel = object_slices[i][0].stop * 4 + object_slices[i][1].stop
    channels_in_each_label.append([start_channel, end_channel])
    print(channels_in_each_label[i])



# PLOTS
# Plot the spread out h.p. data
temp_filtered_spread = pf.spread_data(temp_filtered, electrode_structure, col_spacing=20, row_spacing=2)
plt.figure(1)
plt.plot(temp_filtered_spread[:, :].T)

# Plot the thresholded data spread out
spread_threshold_crossing_regions = pf.spread_data(threshold_crossing_regions, electrode_structure, col_spacing=50, row_spacing=2)
plt.figure(2)
plt.plot(spread_threshold_crossing_regions.T)

# Plot the thresholded data on the probe one time point at a time
pf.scan_through_image_stack(threshold_crossing_regions_on_probe)

