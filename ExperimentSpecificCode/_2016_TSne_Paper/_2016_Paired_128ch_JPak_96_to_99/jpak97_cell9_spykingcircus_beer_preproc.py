
import matplotlib
matplotlib.use('Qt5Agg')
import os
import numpy as np
import IO.ephys as ephys
import matplotlib.pyplot as plt

base_folder = r'F:\JoanaPaired\128ch'
rat = 97
good_cells = '9'
date = '2015-09-03'
data_folder = os.path.join(base_folder + '\\' + date, 'Data')
analysis_folder = os.path.join(base_folder + '\\' + date, 'Analysis')
spyking_circus_folder = os.path.join(analysis_folder, 'spyking_circus')

cell_capture_times = '21_18_47'
spike_thresholds = -2e-4

adc_channel_used = 0
adc_dtype = np.uint16
inter_spike_time_distance = 0.002
amp_gain = 100
num_ivm_channels = 128
amp_dtype = np.uint16

sampling_freq = 30000
high_pass_freq = 500
filtered_data_type = np.float64


num_of_points_in_spike_trig_ivm = 20
num_of_points_for_padding = 2 * num_of_points_in_spike_trig_ivm


raw_data_file_patch = os.path.join(data_folder, 'adc'+date+'T'+cell_capture_times+'.bin')

raw_data_patch = ephys.load_raw_event_trace(raw_data_file_patch, number_of_channels=8,
                                              channel_used=adc_channel_used, dtype=adc_dtype)

raw_data_file_ivm = os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times+'.bin')

raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)


spyking_circus_folder = r'D:\Data\George\Projects\spyking_circus'
raw_data_float32_hp = np.memmap(os.path.join(spyking_circus_folder, 'data_float32_highpassed.dat'), dtype=np.float32, mode='w+', shape=raw_data_ivm.shape())

raw_data_float32 = raw_data_ivm.dataMatrix.copy()

plt.plot(raw_data_float32[:, :100000])
plt.show()






import numpy as np
import h5py as h5
from os.path import join as join

folder = r'D:\Data\George\Projects\spyking_circus'
folder_data = join(folder, 'data')

sampling_freq = 30000

data_float32_hp = np.memmap(join(folder, 'data_float32_hp.dat'), dtype=np.float32)
shape = (128, int(data_float32_hp.shape[0]/128))
data_float32_hp = np.reshape(data_float32_hp, shape, 'F')

channel = 119

juxta_times_hdf5_file = join(folder_data, 'data.beer.hdf5')
h5file = h5.File(juxta_times_hdf5_file, mode='r')
juxta_times = np.array(list(h5file['juxta_spiketimes/elec_0']))


half_spike_length = 0.005
half_spike_length_samples = half_spike_length * sampling_freq
number_of_spikes = len(juxta_times)

extra_spikes_cube = np.empty((number_of_spikes, int(2 * half_spike_length_samples)))
for spike in np.arange(number_of_spikes):
    start = int(juxta_times[spike] - half_spike_length_samples)
    end = int(juxta_times[spike] + half_spike_length_samples)
    extra_spikes_cube[spike, :] = data_float32_hp[channel, start:end]


time_axis = np.arange(-half_spike_length, half_spike_length, 1/sampling_freq)

plt.plot(time_axis, extra_spikes_cube.T)



folder_result = r'D:\Data\George\Projects\spyking_circus\data'
beer_filename = join(folder_result, 'data.beer.hdf5')
bfile = h5.File(beer_filename, 'r', libver='latest')
confusion_matrices = bfile.get('confusion_matrices')[:]
bfile.close()


fprs = [M[1, 0] / (M[1, 0] + M[1, 1]) for M in confusion_matrices]
tprs = [M[0, 0] / (M[0, 0] + M[0, 1]) for M in confusion_matrices]
# Add false positive rates and true positive rates endpoints.
fprs = np.array([1.0] + fprs + [0.0])
tprs = np.array([1.0] + tprs + [0.0])


plt.semilogx(fprs,tprs)
plt.plot(fprs, tprs)

# Process for throwing away the first few and last few data points of the data.dat binary file
file = r'F:\JoanaPaired\128ch\2015-09-03\Analysis\spyking_circus\data.dat'
num_of_points_to_clean = 5000
data = np.memmap(file, np.float32)
data = np.reshape(data, (128, int(data.shape[0]/128)), 'F')
averages_begin = np.median(data[:, :num_of_points_to_clean], axis=1)
averages_begin = np.tile(averages_begin, (num_of_points_to_clean, 1)).T
data[:, :num_of_points_to_clean] = averages_begin

plt.plot(data[:, :10000].T)

num_of_points_to_clean = 2000
averages_end = np.median(data[:, -num_of_points_to_clean:], axis=1)
averages_end = np.tile(averages_end, (num_of_points_to_clean, 1)).T
data[:, -num_of_points_to_clean:] = averages_end

plt.plot(data[:, -10000:].T)


