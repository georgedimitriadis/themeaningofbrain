import matplotlib.pyplot as plt
import mne.filter as filters
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from matplotlib import mlab
import os
import scipy.integrate as integrate

import IO.ephys as ephys
import itertools
import os
import warnings

import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import scipy.signal as signal
import scipy.stats as stats

import IO.ephys as ephys
import os
import numpy as np
import BrainDataAnalysis.timelocked_analysis_functions as tf
import IO.ephys as ephys
import scipy.signal as signal

#------------------------------------------------------------------------------------------------------------------------

#128ch probes-----------------------------------------------------------------------------------------------------------


#SWC data brain 2016-08-17

analysis_folder = r'E:\Paper Impedance\128chNeuroseeker\SWC\2016_08_17\Rec1'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2016-08-17T13_26_10.bin')


analysis_folder = r'E:\Paper Impedance\128chNeuroseeker\SWC\2016_08_17\Rec2'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2016-08-17T16_07_34.bin')


#SWC saline this is not correct
analysis_folder = r'E:\Paper Impedance\128chNeuroseeker\Noise\2016_08_26'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + '128ch_new_RMS_saline.bin')

#CCU data brain 2015-08-28

analysis_folder = r'E:\Paper Impedance\128chNeuroseeker\CCU\2015_08_28\pair2.2'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2015-08-28T20_15_45.bin')

#CCU data brain 2015-09-23

analysis_folder = r'E:\Paper Impedance\128chNeuroseeker\CCU\2015_09_03\pair9.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2015-09-03T21_18_47.bin')

#SWC mice head fixed

analysis_folder = r'E:\Paper Impedance\128chNeuroseeker\SWC\2016_08_12\16_53_27'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2016-08-12T16_53_27.bin')



amp_dtype = np.uint16
Probe_y_digitization = 32768
num_ivm_channels = 128
sampling_freq = 30000
filtered_data_type = np.float64
voltage_step_size = 0.195e-6
scale_uV = 1000000


raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_unfiltered_uV = (temp_unfiltered - Probe_y_digitization) * scale_uV * voltage_step_size
high_pass_freq = 250

def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=100.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)

tfinal = np.ceil((temp_unfiltered_uV.shape[1])/2)
temp_filtered_uV = highpass(temp_unfiltered_uV[:,0:tfinal], F_HIGH = (sampling_freq/2)*0.95, sampleFreq = sampling_freq, passFreq = high_pass_freq)


#Protocol2 to calculate the stdv from noise RMS
# RMS noise level for all channels

def RMS_calculation(data):

    RMS = np.zeros(num_ivm_channels)

    for i in range(num_ivm_channels):
        RMS[i] = np.sqrt((1/len(data[i]))*np.sum(data[i]**2))

    return RMS

noise_rms = RMS_calculation(temp_filtered_uV)
noise_rms_average = np.average(noise_rms)
noise_rms_stdv = stats.sem(noise_rms)

print('#------------------------------------------------------')
print('RMS:'+ str(noise_rms))
print('RMS_average:'+ str(noise_rms_average))
print('RMS_average_stdv:'+ str(noise_rms_stdv))
print('#------------------------------------------------------')

filename_RMS = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_RMS' + '.npy')
np.save(filename_RMS, noise_rms)


#Protocol1 to calculate the stdv from noise MEDIAN

noise_median = np.median(np.abs(temp_filtered_uV)/0.6745, axis=1)
noise_median_average = np.average(noise_median)
noise_median_stdv = stats.sem(noise_median)
print('#------------------------------------------------------')
print('Noise_Median:'+ str(noise_median))
print('Noise_Median_average:'+ str(noise_median_average))
print('Noise_Median_stdv:'+ str(noise_median_stdv))
print('#------------------------------------------------------')

filename_Median = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_Median' + '.npy')
np.save(filename_Median, noise_median)


#compare SWC-ketamine vs CCU-ketamine vs SWC headfixed

num_samples = temp_filtered_uV.shape[1]
sample_axis= np.arange(-(num_samples/2),(num_samples/2))
time_axis = sample_axis/sampling_freq
yoffset =0
channel = 0
scale_ms =1000
#plt.figure()
plt.plot(time_axis*scale_ms , temp_filtered_uV[channel,:].T)
plt.ylim(np.min(temp_filtered_uV[channel,:])-yoffset, np.max(temp_filtered_uV[channel,:])+yoffset)
plt.ylabel('Voltage (\u00B5V)', fontsize=20)
plt.xlabel('Time (ms)',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)




#128ch-----------------------------------------------50kOhm

# Calculate ELECTRONIC noise (saline solution, resistors...for all frequencies the data was saved int16)
num_ivm_channels = 128
amp_dtype = np.int16
#amp_dtype = np.uint16
sampling_freq = 30000
filtered_data_type = np.float64
voltage_step_size = 0.195e-6
scale_uV = 1000000

#Open Data-------------------------------------------
raw_data_file_ivm =r'F:\DataKilosort\2015_09_03_Cell9_0_128chand256PCA\amplifier2015-09-03T21_18_47.bin'

raw_data_file_ivm =r'F:\DataKilosort\2016_08_12_amplifier2016-08-12T16_53_27\amplifier2016-08-12T16_53_27.bin'

raw_data_file_ivm =r'H:\2016_08_26\128ch_new_RMS_saline.bin'

raw_data_file_ivm = r'F:\DataKilosort\2016_08_17\amplifier2016-08-17T13_26_10\amplifier2016-08-17T13_26_10.bin'


raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)


#SPD spectrum---------------------------------------------------------
#top electrodes bottom electrodes
top =[47,	45,	43,	41,	1, 61,	57, 36,	34,	32,	30,	28,	26,	24,	22,	20, 49, 51, 53, 55, 2, 62, 58, 4, 6, 8, 10, 12, 14, 21, 19, 16, 46, 44, 42, 40, 38, 63, 59,39, 37, 35, 33, 31, 29, 27, 25, 23, 48, 50, 52, 54, 56, 0, 60, 3, 5, 7, 9, 11, 13, 15, 18]
bottom = [103,	101, 99,	97,	95,	93,	91,	89,	87,	70,	66,	84,	82,	80,	108,110,106, 104, 115, 117, 119, 121, 123, 125, 127, 71, 67, 74, 76, 78, 114, 112,102, 100, 98, 96, 94, 92, 90, 88, 86, 72, 68, 65, 85, 83, 81, 111,109, 107, 105, 116, 118, 120, 122, 124, 126, 73, 69, 64, 75, 77, 79, 113]

plt.figure()
for i in np.arange(len(top)):
    Pxx_dens,f = mlab.psd((temp_unfiltered[top[i],:])*scale_uV * voltage_step_size, sampling_freq, sampling_freq)
    plt. semilogy(f, Pxx_dens,color='r',linewidth=1)
    plt.semilogx()
    plt.ylim([1e-5,1000000])
    plt.xlim([2,17000])
    #plt.ylim([0.5e-3,100])
    #plt.xlim([1,6000])
    plt.xlabel('frequency(Hz)')
    plt.ylabel('PSD (uV^2/Hz)')

for i in np.arange(len(bottom)):
    Pxx_dens,f = mlab.psd(temp_unfiltered[bottom[i],:]* scale_uV * voltage_step_size, sampling_freq, sampling_freq)
    plt. semilogy(f, Pxx_dens,color='b',linewidth=1)
    plt.semilogx()
    plt.ylim([1e-5,1000000])
    plt.xlim([2,17000])
    plt.xlabel('frequency(Hz)')
    plt.ylabel('PSD (uV^2/Hz)')


#128channels

plt.figure()
power=[]
for i in np.arange(num_ivm_channels):
    temp_filtered_uV = temp_unfiltered[i,:] * scale_uV *voltage_step_size
    Pxx_dens,f = mlab.psd(temp_unfiltered[i,:], sampling_freq, sampling_freq)
    plt. semilogy(f, Pxx_dens,color='b',linewidth=1)
    plt.semilogx()
    plt.ylim([1e-5,1000000])
    plt.xlim([2,17000])
    plt.xlabel('frequency(Hz)')
    plt.ylabel('PSD (uV^2/Hz)')
    power = np.append(power, Pxx_dens)

power = np.reshape(power,[num_ivm_channels, power.shape[0]/num_ivm_channels])
power_average = np.average(power, axis=0)
plt.figure()
plt.semilogy(f, power_average,color='b',linewidth=1)
plt.semilogx()
plt.ylim([1e-5,10000])
plt.xlim([1,300])

#Highpass 128ch
#temp_filtered_uV = temp_unfiltered[:, 0:3000000] * scale_uV * voltage_step_size

temp_filtered_uV = temp_unfiltered * scale_uV * voltage_step_size

def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=250.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)


temp_filtered_uV = highpass(temp_filtered_uV[:, 0:5000000])* scale_uV * voltage_step_size



#RMS amplitude----------------------------------------------
#all channels

stdvs_ch = np.median(np.abs(temp_filtered_uV[:])/0.6745, axis=1)
stdv_ch_average = np.average(stdvs_ch)
stdv_ch_stdv = stats.sem(stdvs_ch)
print('#------------------------------------------------------')
print('RMS_ch:'+ str(stdvs_ch))
print('RMS_ch_average:'+ str(stdv_ch_average))
print('RMS_ch_stdv:' + str(stdv_ch_stdv))
print('#------------------------------------------------------')

#plot all signals---------------------------------
num_samples=temp_filtered_uV.shape[1]
sample_axis= np.arange(0,num_samples)
time_axis= sample_axis/30000

plt.figure()#3sec
plt.plot(time_axis[0:90000],temp_filtered_uV[:, 30000:120000].T, color='b', label= 'Pristine')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(-1000,500)



plt.figure()#3sec
plt.plot(time_axis[0:90000],temp_filtered_uV[:, 0:90000].T, color='b', label= 'Pristine')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(-1000,500)