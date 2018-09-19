
import os
import numpy as np
import BrainDataAnalysis.timelocked_analysis_functions as tf
import IO.ephys as ephys
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt


num_of_points_in_spike_trig_ivm = 128
num_of_points_for_padding = 2 * num_of_points_in_spike_trig_ivm

inter_spike_time_distance = 30
amp_gain = 1000
num_ivm_channels = 32
amp_dtype = np.float32

sampling_freq = 30000
high_pass_freq = 100
filtered_data_type = np.float64




raw_data_file_ivm = r'F:\Materials paper w Pedro Baiao\Results_ATLAS\atlas\2013_07_19_passive probes\Surgery day\21.57h_Aligned Site 30\recordings0.bin'

raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)


def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)


window = 0
window_size_secs = 60
filtered_data_type = np.float64
sampling_freq = 30000
high_pass_freq = 100
window_size = int(window_size_secs * sampling_freq)


temp_unfiltered = raw_data_ivm.dataMatrix
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_filtered = highpass(temp_unfiltered)


num_samples=temp_filtered.shape[1]
sample_axis= np.arange(0,num_samples)
time_axis= sample_axis/30000



small_electrodes = [29,30,7,5,1,26,25,31,6,13]
big_electrodes = [28,18,2,3,22,27,4,9,21,23,8,10,19,20,11,12]
plt.figure()

for i in np.arange(1,len(small_electrodes)+1):
    plt.subplot(2,5,i)
    plt.plot(time_axis[:], temp_filtered[small_electrodes[i-1], :].T)
    plt.ylim(-800,200)

plt.figure()
for i in np.arange(1,len(big_electrodes)+1):
    plt.subplot(4,4,i)
    plt.plot(time_axis[:], temp_filtered[big_electrodes[i-1], :].T)
    plt.ylim(-800,200)


#plot overlaid
plt.figure()
for i in np.arange(0,len(big_electrodes)):
    plt.plot(time_axis[:], temp_filtered[big_electrodes[i], :].T+ 200*i)
    #plt.ylim(-800,200)

#plt.figure()
for i in np.arange(0,len(small_electrodes)):
    plt.plot(time_axis[:], temp_filtered[small_electrodes[i], :].T - 200-200*i)
    #plt.ylim(-800,200)


sheme_triangle= [29,30,7,26,25,31,28,18,2]
plt.figure()
for i in np.arange(0,len(sheme_triangle)):
    plt.plot(time_axis[:], temp_filtered[sheme_triangle[i], :].T+ 200*i)







plt.figure(); plt.plot(time_axis[:], temp_filtered[2, :].T, color='g', label= 'Pristine')
plt.plot(time_axis[0:30000], temp_filtered[22, 0:30000].T , color='k', label = 'PEDOT')
plt.ylabel('Voltage (\u00B5V)', fontsize=20)
plt.xlabel('Time (s)',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend()
plt.ylim(-500,200)
