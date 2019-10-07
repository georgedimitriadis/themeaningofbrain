import os
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mne.filter as filters
import numpy as np
import scipy.stats as stats
from matplotlib import mlab
import IO.ephys as ephys

#File names-------------------------------------------------------------------------------------------------------------
#NEURONEXUS 5
#All channels pristine

#recording 2014-10-10 Pair 1.0
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_10_10\Pair1.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-10-10T17_30_04.bin')

#recording 2014-10-17 Pair 1.0
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_10_17\Pair1.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-10-17T16_46_02.bin')

#recording 2014-10-17 Pair 1.1
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_10_17\Pair1.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-10-17T17_12_27.bin')

#recording 2014-10-17 Pair2.0
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_10_17\Pair2.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-10-17T18_19_09.bin')

#recording 2014-10-17 Rec1_layer2-3
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_10_17\Rec1_layer2_3'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-10-17T13_06_57.bin')

#recording 2015-05-05 SC
analysis_folder = r'Z:\j\Joana Neto\Backup_2017_28_06\PCDisk\Paper Impedance\Neuronexus5\NoisePower\Electronic noise_2015_ 05_05\SC-headstage'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2015-05-05T17_20_09.bin')

#recording 2015-05-05 1kOhm
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Electronic noise_2015_ 05_05\1kOhm'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2015-05-11T12_38_44.bin')

#recording 2015-05-05 100kOhm
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Electronic noise_2015_ 05_05\97kOhm'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2015-05-05T18_39_02.bin')

#recording 2015-05-05 1MOhm
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Electronic noise_2015_ 05_05\1MOhm'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2015-05-05T18_32_50.bin')

# recording 2015-05-05 10MOhm
analysis_folder =r'Z:\j\Joana Neto\Backup_2017_28_06\PCDisk\Paper Impedance\Neuronexus5\NoisePower\Electronic noise_2015_ 05_05\9.9MOhm'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2015-05-05T18_36_49.bin')


#Open Data--------------------------------------------------------------------------------------------------------------
amp_dtype = np.uint16
Probe_y_digitization = 32768
num_ivm_channels = 32
sampling_freq = 30000
filtered_data_type = np.float64
voltage_step_size = 0.195e-6
scale_uV = 1000000

raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_unfiltered_uV = (temp_unfiltered - Probe_y_digitization) * scale_uV * voltage_step_size


#High-pass filtered data------------------------------------------------------------------------------------------------
high_pass_freq = 250
def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=100.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)


temp_filtered_uV = highpass(temp_unfiltered_uV, F_HIGH = (sampling_freq/2)*0.95, sampleFreq = sampling_freq, passFreq = high_pass_freq)


#Protocol to calculate noise RMS----------------------------------------------------------------------------------------
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


#Protocol1 to calculate noise MEDIAN------------------------------------------------------------------------------------
# RMS noise level for all channels
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


#Electronic noise w different resistors(1k, 100k, 1M and 10MOhm); some channels are not connected-----------------------
#Remove this channels

#Protocol to calculate noise RMS
def RMS_calculation(data):

    RMS = np.zeros(num_ivm_channels)

    for i in range(num_ivm_channels):
        RMS[i] = np.sqrt((1/len(data[i]))*np.sum(data[i]**2))

    return RMS

noise_rms = RMS_calculation(temp_filtered_uV)
noise_rms_average = np.average(noise_rms[np.where(noise_rms < 4)])
noise_rms_stdv = stats.sem(noise_rms[np.where(noise_rms < 4)])

print('#------------------------------------------------------')
print('RMS:'+ str(noise_rms))
print('RMS_average:'+ str(noise_rms_average))
print('RMS_average_stdv:'+ str(noise_rms_stdv))
print('#------------------------------------------------------')

filename_RMS = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_RMS' + '.npy')
np.save(filename_RMS, noise_rms)


#Protocol1 to calculate noise MEDIAN
noise_median = np.median(np.abs(temp_filtered_uV)/0.6745, axis=1)
noise_median_average = np.average(noise_median[np.where(noise_rms < 4)])
noise_median_stdv = stats.sem(noise_median[np.where(noise_rms < 4)])
print('#------------------------------------------------------')
print('Noise_Median:'+ str(noise_median))
print('Noise_Median_average:'+ str(noise_median_average))
print('Noise_Median_stdv:'+ str(noise_median_stdv))
print('#------------------------------------------------------')

filename_Median = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_Median' + '.npy')
np.save(filename_Median, noise_median)


#File names-------------------------------------------------------------------------------------------------------------
# 16 channels pedot and 16 channels pristine

#recording 2014-11-25  Pair3.0
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_25\pair3.0'
#raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-25T23_00_08.bin')
raw_data_file_ivm= r"K:\Neuronexus32ch\2014-11-25\Data\amplifier2014-11-25T23_00_08.bin"

#recording 2014-11-25  Pair3.0 after CAR 1Hz
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_25\pair3.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-25T23_00_08_bonsaiCAR1Hz_allchs.bin')
amp_dtype = np.int16

#recording 2014-11-25  Pair3.0 after CAR 0Hz
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_25\pair3.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-25T23_00_08_bonsaiCAR0Hz_allchs.bin')
amp_dtype = np.int16

#recording 2014-11-25  Pair2.0
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_25\pair2.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-25T22_44_57.bin')

#recording 2014-11-25  Pair1.0
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_25\pair1.0'
#raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-25T21_27_13.bin')
raw_data_file_ivm= r"K:\Neuronexus32ch\2014-11-25\Data\amplifier2014-11-25T21_27_13.bin"

#recording 2014-11-25  Rec0
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_25\rec0'
#raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-25T20_32_48.bin')
raw_data_file_ivm= r"K:\Neuronexus32ch\2014-11-25\Data\amplifier2014-11-25T20_32_48.bin"

#recording 2014-11-13  Pair1.0
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_13\Pair1.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-13T19_01_55.bin')

#recording 2014-11-13  Pair3.0
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_13\pair3.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-13T18_48_11.bin')

#recording 2014-11-13 rec14_59_40
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_13\14_59_40'
#raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-13T14_59_40.bin')
raw_data_file_ivm=r"K:\Neuronexus32ch\2014-11-13\Data\amplifier2014-11-13T14_59_40.bin"

#recording 2014-11-13 rec15_35_31
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_13\15_35_31'
#raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-13T15_35_31.bin')
raw_data_file_ivm=r"K:\Neuronexus32ch\2014-11-13\Data\amplifier2014-11-13T15_35_31.bin"

#recording 2014-11-13 rec18_05_50
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_13\18_05_50'
#raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-13T18_05_50.bin')
raw_data_file_ivm=r"K:\Neuronexus32ch\2014-11-13\Data\amplifier2014-11-13T18_05_50.bin"

#recording 2014-11-13 rec21_05_14
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_13\21_05_14'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-13T21_05_14.bin')

#recording 2014-11-13  rec1_extra
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_13\rec1_extra'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-13T14_25_42.bin')

#recording 2014-11-13  rec2_extra x juxta insertion
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_13\rec2_extrawjuxta'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-13T14_59_40.bin')

#recording 2015-04-24  cell 1
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2015_04_24\cell1'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2015-04-24T15_24_49.bin')

#recording 2015-05-11  saline w juxta
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Saline noise_2015_05_11\RMS saline_probe and juxta'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2015-05-11T12_16_44.bin')

#recording 2015-05-11  saline w/out juxta
analysis_folder = r'Z:\j\Joana Neto\Backup_2017_28_06\PCDisk\Paper Impedance\Neuronexus5\Noise\Saline noise_2015_05_11\RMS saline_probe'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2015-05-11T11_59_54.bin')


#Open Data--------------------------------------------------------------------------------------------------------------
amp_dtype = np.uint16
num_ivm_channels = 32
sampling_freq = 30000
filtered_data_type = np.float64
voltage_step_size = 0.195e-6
scale_uV = 1000000
Probe_y_digitization = 32768

raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_unfiltered_uV = (temp_unfiltered - Probe_y_digitization) * scale_uV * voltage_step_size
high_pass_freq = 250



#High-pass filtered data------------------------------------------------------------------------------------------------
def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=100.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)


temp_filtered_uV = highpass(temp_unfiltered_uV, F_HIGH = (sampling_freq/2)*0.95, sampleFreq = sampling_freq, passFreq = high_pass_freq)


#Plot-------------------------------------------------------------------------------------------------------------------
plt.figure()
plt.plot(temp_unfiltered_uV[22,6000000:6150000])
plt.plot(temp_filtered_uV[22,6000000:6150000])

plt.figure()
plt.plot(temp_unfiltered_uV[22,:])
plt.plot(temp_filtered_uV[22,:])

#Protocol to calculate noise RMS----------------------------------------------------------------------------------------
#RMS noise level for pristine vs pedot channels
channels_pedot= [22,29,3,23,18,4,17,19,5,16,20,6,30,21,7,31]
channels_pristine= [2,9,28,13,8,27,12,14,26,11,15,25,10,1,24,0]

def RMS_calculation(data):

    RMS = np.zeros(len(channels_pedot))

    for i in range(len(channels_pedot)):
        RMS[i] = np.sqrt((1/len(data[i]))*np.sum(data[i]**2))

    return RMS

noise_rms_pedot = RMS_calculation(temp_filtered_uV[channels_pedot])
noise_rms_pristine = RMS_calculation(temp_filtered_uV[channels_pristine])

noise_rms_average_pedot = np.average(noise_rms_pedot)
noise_rms_average_pristine = np.average(noise_rms_pristine)

noise_rms_stdv_pedot = stats.sem(noise_rms_pedot)
noise_rms_stdv_pristine = stats.sem(noise_rms_pristine)

print('#------------------------------------------------------')
print('RMSpedot:'+ str(noise_rms_pedot))
print('RMS_averagepedot:'+ str(noise_rms_average_pedot))
print('RMS_average_stdvpedot:'+ str(noise_rms_stdv_pedot))

print('RMSpristine:'+ str(noise_rms_pristine))
print('RMS_averagepridtine:'+ str(noise_rms_average_pristine))
print('RMS_average_stdvpristine:'+ str(noise_rms_stdv_pristine))
print('#------------------------------------------------------')

filename_RMS = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_RMS_PEDOT' + '.npy')
np.save(filename_RMS, noise_rms_pedot)
filename_RMS = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_RMS_pristine' + '.npy')
np.save(filename_RMS, noise_rms_pristine)


#Protocol to calculate noise MEDIAN-------------------------------------------------------------------------------------
noise_medianpedot = np.median(np.abs(temp_filtered_uV[channels_pedot])/0.6745, axis=1)
noise_medianpristine = np.median(np.abs(temp_filtered_uV[channels_pristine])/0.6745, axis=1)

noise_median_averagepedot = np.average(noise_medianpedot)
noise_median_averagepristine = np.average(noise_medianpristine)

noise_median_stdvpedot = stats.sem(noise_medianpedot)
noise_median_stdvpristine = stats.sem(noise_medianpristine)

print('#------------------------------------------------------')
print('Noise_MedianPristine:'+ str(noise_medianpristine))
print('Noise_Median_averagePristine:'+ str(noise_median_averagepristine))
print('Noise_Median_stdvPristine:'+ str(noise_median_stdvpristine))
print('Noise_MedianPedot:'+ str(noise_medianpedot))
print('Noise_Median_averagePedot:'+ str(noise_median_averagepedot))
print('Noise_Median_stdvPedot:'+ str(noise_median_stdvpedot))
print('#------------------------------------------------------')

filename_Median = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_Median_pristine' + '.npy')
np.save(filename_Median, noise_medianpristine)

filename_Median = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_Median_PEDOT' + '.npy')
np.save(filename_Median, noise_medianpedot)


#File names-------------------------------------------------------------------------------------------------------------
#NEURONEXUS 6

#recording 2017-02-02  saline
analysis_folder = r'E:\Paper Impedance\Neuronexus6\Noise\RMSSaline'
#raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2017-02-02T10_30_29.bin')
raw_data_file_ivm = r"Z:\j\Joana Neto\Backup_2017_28_06\PCDisk\Paper Impedance\Neuronexus6\Noise\Saline\amplifier2017-02-02T10_30_29.bin"

#recording 2017-02-02  rec1
analysis_folder = r'E:\Paper Impedance\Neuronexus6\Noise\rec1'
#raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2017-02-02T14_38_11.bin')
raw_data_file_ivm= r"K:\Neuronexus32ch\2017-02-02\Data\amplifier2017-02-02T14_38_11.bin"

#recording 2017-02-02  rec2
analysis_folder = r'E:\Paper Impedance\Neuronexus6\Noise\rec2'
#raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2017-02-02T15_03_44.bin')
raw_data_file_ivm= r"K:\Neuronexus32ch\2017-02-02\Data\amplifier2017-02-02T15_03_44.bin"

#recording 2017-02-02  rec3
analysis_folder = r'E:\Paper Impedance\Neuronexus6\Noise\rec3'
#raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2017-02-02T15_49_35.bin')#
#raw_data_file_ivm= r'Z:\j\Joana Neto\Neuronexus32ch_clusters\2017-02-02\Data\amplifier2017-02-02T15_49_35.bin'
raw_data_file_ivm= r"K:\Neuronexus32ch\2017-02-02\Data\amplifier2017-02-02T15_49_35.bin"

#recording 2017-02-02  rec4
analysis_folder = r'E:\Paper Impedance\Neuronexus6\Noise\rec4'
#raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2017-02-02T16_57_16.bin')
raw_data_file_ivm=r"K:\Neuronexus32ch\2017-02-02\Data\amplifier2017-02-02T16_57_16.bin"

#recording 2017-02-02  rec5
analysis_folder = r'E:\Paper Impedance\Neuronexus6\Noise\rec5'
#raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2017-02-02T17_18_46.bin')
raw_data_file_ivm=r"Z:\j\Joana Neto\Backup_2017_28_06\PCDisk\Paper Impedance\Neuronexus6\Noise\rec5\amplifier2017-02-02T17_18_46.bin"


#Open Data--------------------------------------------------------------------------------------------------------------
amp_dtype = np.uint16
Probe_y_digitization = 32768
num_ivm_channels = 32
sampling_freq = 20000
filtered_data_type = np.float64
voltage_step_size = 0.195e-6
scale_uV = 1000000

raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_unfiltered_uV = (temp_unfiltered - Probe_y_digitization) * scale_uV * voltage_step_size
high_pass_freq = 250


#High-pass filtered data------------------------------------------------------------------------------------------------
def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=100.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)


temp_filtered_uV = highpass(temp_unfiltered_uV, F_HIGH = (sampling_freq/2)*0.95, sampleFreq = sampling_freq, passFreq = high_pass_freq)




#Protocol to calculate noise RMS----------------------------------------------------------------------------------------
#RMS noise level for pristine vs pedot channels
channels_pedot= [22,29,3,23,18,4,17,19,5,16,20,6,30,21,7,31]

channels_pristine= [2,9,28,13,8,27,12,14,26,11,15,25,10,1,24,0]

def RMS_calculation(data):

    RMS = np.zeros(len(channels_pedot))

    for i in range(len(channels_pedot)):
        RMS[i] = np.sqrt((1/len(data[i]))*np.sum(data[i]**2))

    return RMS

noise_rms_pedot = RMS_calculation(temp_filtered_uV[channels_pedot])
noise_rms_pristine = RMS_calculation(temp_filtered_uV[channels_pristine])

noise_rms_average_pedot = np.average(noise_rms_pedot)
noise_rms_average_pristine = np.average(noise_rms_pristine)

noise_rms_stdv_pedot = stats.sem(noise_rms_pedot)
noise_rms_stdv_pristine = stats.sem(noise_rms_pristine)

print('#------------------------------------------------------')
print('RMSpedot:'+ str(noise_rms_pedot))
print('RMS_averagepedot:'+ str(noise_rms_average_pedot))
print('RMS_average_stdvpedot:'+ str(noise_rms_stdv_pedot))

print('RMSpristine:'+ str(noise_rms_pristine))
print('RMS_averagepridtine:'+ str(noise_rms_average_pristine))
print('RMS_average_stdvpristine:'+ str(noise_rms_stdv_pristine))
print('#------------------------------------------------------')

filename_RMS = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_RMS_PEDOT' + '.npy')
np.save(filename_RMS, noise_rms_pedot)
filename_RMS = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_RMS_pristine' + '.npy')
np.save(filename_RMS, noise_rms_pristine)


#Protocol to calculate noise MEDIAN-------------------------------------------------------------------------------------
#MEDIAN noise level for pristine vs pedot channels
noise_medianpedot = np.median(np.abs(temp_filtered_uV[channels_pedot])/0.6745, axis=1)
noise_medianpristine = np.median(np.abs(temp_filtered_uV[channels_pristine])/0.6745, axis=1)

noise_median_averagepedot = np.average(noise_medianpedot)
noise_median_averagepristine = np.average(noise_medianpristine)

noise_median_stdvpedot = stats.sem(noise_medianpedot)
noise_median_stdvpristine = stats.sem(noise_medianpristine)


print('#------------------------------------------------------')
print('Noise_MedianPristine:'+ str(noise_medianpristine))
print('Noise_Median_averagePristine:'+ str(noise_median_averagepristine))
print('Noise_Median_stdvPristine:'+ str(noise_median_stdvpristine))
print('Noise_MedianPedot:'+ str(noise_medianpedot))
print('Noise_Median_averagePedot:'+ str(noise_median_averagepedot))
print('Noise_Median_stdvPedot:'+ str(noise_median_stdvpedot))
print('#------------------------------------------------------')

filename_Median = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_Median_Pristine' + '.npy')
np.save(filename_Median, noise_medianpristine)

filename_Median = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_Median_PEDOT' + '.npy')
np.save(filename_Median, noise_medianpedot)


#File names-------------------------------------------------------------------------------------------------------------
#lori32ch probe
#saline

analysis_folder = r"Z:\j\Joana Neto\Backup_2017_28_06\PCdisk2\Materials paper w Pedro Baiao\Results_ChronicLori32ch\2016-10-26"
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'lori_RMS_saline_int162016-11-01T19_59_18.bin')


#Open Data--------------------------------------------------------------------------------------------------------------
amp_dtype = np.int16
Probe_y_digitization = 32768
num_ivm_channels = 32
sampling_freq = 30000
filtered_data_type = np.float64
voltage_step_size = 0.195e-6
scale_uV = 1000000

raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_unfiltered_uV = (temp_unfiltered - Probe_y_digitization) * scale_uV * voltage_step_size
high_pass_freq = 250



#High-pass filtered data------------------------------------------------------------------------------------------------
def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=100.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)


temp_filtered_uV = highpass(temp_unfiltered_uV, F_HIGH = (sampling_freq/2)*0.95, sampleFreq = sampling_freq, passFreq = high_pass_freq)




#Save npy to txt to open in Origin--------------------------------------------------------------------------------------

#lori32ch probe---------------------------------------------------------------------------------------------------------
folder = r'Z:\j\Joana Neto\Backup_2017_28_06\PCdisk2\Materials paper w Pedro Baiao\Results_ChronicLori32ch\2016-10-26'


channels = 'Pristine'
#RMS
noisefile = r'250noise_RMS_pristine.npy'
RMS_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pristine = np.load(RMS_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline_pristine.txt"),"w")
string = (str(RMS_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pristine.npy'
median_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pristine = np.load(median_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline_pristine.txt"),"w")
string = (str(median_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


channels = 'PEDOT'
#RMS
noisefile = r'250noise_RMS_PEDOT.npy'
RMS_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pedot = np.load(RMS_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline_PEDOT.txt"),"w")
string = (str(RMS_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pedot.npy'
median_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pedot = np.load(median_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline_PEDOT.txt"),"w")
string = (str(median_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()



#Neuronexus5 w pippete in saline----------------------------------------------------------------------------------------
folder = r'E:\Paper Impedance\Neuronexus5\Noise\Saline noise_2015_05_11\RMS saline_probe and juxta'


channels = 'Pristine'
#RMS
noisefile = r'250noise_RMS_pristine.npy'
RMS_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pristine = np.load(RMS_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline.txt"),"w")
string = (str(RMS_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pristine.npy'
median_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pristine = np.load(median_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline.txt"),"w")
string = (str(median_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


channels = 'PEDOT'
#RMS
noisefile = r'250noise_RMS_PEDOT.npy'
RMS_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pedot = np.load(RMS_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline.txt"),"w")
string = (str(RMS_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pedot.npy'
median_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pedot = np.load(median_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline.txt"),"w")
string = (str(median_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


#Neuronexus6 in saline--------------------------------------------------------------------------------------------------
folder = r'E:\Paper Impedance\Neuronexus6\Noise\Saline'

channels = 'Pristine'
#RMS
noisefile = r'250noise_RMS_pristine.npy'
RMS_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pristine = np.load(RMS_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline.txt"),"w")
string = (str(RMS_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pristine.npy'
median_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pristine = np.load(median_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline.txt"),"w")
string = (str(median_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


channels = 'PEDOT'
#RMS
noisefile = r'250noise_RMS_PEDOT.npy'
RMS_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pedot = np.load(RMS_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline.txt"),"w")
string = (str(RMS_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pedot.npy'
median_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pedot = np.load(median_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline.txt"),"w")
string = (str(median_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()



#Neuronexus6 rec3 15_49_35----------------------------------------------------------------------------------------------
folder = r'E:\Paper Impedance\Neuronexus6\Noise\rec3'

channels = 'Pristine'
#RMS
noisefile = r'250noise_RMS_pristine.npy'
RMS_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pristine = np.load(RMS_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline_pristine.txt"),"w")
string = (str(RMS_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pristine.npy'
median_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pristine = np.load(median_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline_pristine.txt"),"w")
string = (str(median_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


channels = 'PEDOT'
#RMS
noisefile = r'250noise_RMS_PEDOT.npy'
RMS_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pedot = np.load(RMS_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline_PEDOT.txt"),"w")
string = (str(RMS_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pedot.npy'
median_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pedot = np.load(median_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline_PEDOT.txt"),"w")
string = (str(median_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


#Neuronexus6 rec2 15_03_44----------------------------------------------------------------------------------------------
folder = r'E:\Paper Impedance\Neuronexus6\Noise\rec2'

channels = 'Pristine'
#RMS
noisefile = r'250noise_RMS_pristine.npy'
RMS_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pristine = np.load(RMS_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline_pristine.txt"),"w")
string = (str(RMS_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pristine.npy'
median_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pristine = np.load(median_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline_pristine.txt"),"w")
string = (str(median_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


channels = 'PEDOT'
#RMS
noisefile = r'250noise_RMS_PEDOT.npy'
RMS_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pedot = np.load(RMS_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline_PEDOT.txt"),"w")
string = (str(RMS_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pedot.npy'
median_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pedot = np.load(median_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline_PEDOT.txt"),"w")
string = (str(median_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()



#Neuronexus6 rec1 14_38_11----------------------------------------------------------------------------------------------
folder = r'E:\Paper Impedance\Neuronexus6\Noise\rec1'

channels = 'Pristine'
#RMS
noisefile = r'250noise_RMS_pristine.npy'
RMS_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pristine = np.load(RMS_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline_pristine.txt"),"w")
string = (str(RMS_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pristine.npy'
median_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pristine = np.load(median_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline_pristine.txt"),"w")
string = (str(median_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


channels = 'PEDOT'
#RMS
noisefile = r'250noise_RMS_PEDOT.npy'
RMS_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pedot = np.load(RMS_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline_PEDOT.txt"),"w")
string = (str(RMS_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_PEDOT.npy'
median_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pedot = np.load(median_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline_PEDOT.txt"),"w")
string = (str(median_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()



#Neuronexus6 rec4 amplifier2017-02-02T16_57_16--------------------------------------------------------------------------
folder = r'E:\Paper Impedance\Neuronexus6\Noise\rec4'

channels = 'Pristine'
#RMS
noisefile = r'250noise_RMS_pristine.npy'
RMS_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pristine = np.load(RMS_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline_pristine.txt"),"w")
string = (str(RMS_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pristine.npy'
median_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pristine = np.load(median_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline_pristine.txt"),"w")
string = (str(median_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


channels = 'PEDOT'
#RMS
noisefile = r'250noise_RMS_PEDOT.npy'
RMS_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pedot = np.load(RMS_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline_PEDOT.txt"),"w")
string = (str(RMS_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_PEDOT.npy'
median_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pedot = np.load(median_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline_PEDOT.txt"),"w")
string = (str(median_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


#Neuronexus6 rec5 amplifier2017-02-02T17_18_46--------------------------------------------------------------------------
folder = r'E:\Paper Impedance\Neuronexus6\Noise\rec5'

channels = 'Pristine'
#RMS
noisefile = r'250noise_RMS_pristine.npy'
RMS_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pristine = np.load(RMS_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline_pristine.txt"),"w")
string = (str(RMS_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pristine.npy'
median_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pristine = np.load(median_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline_pristine.txt"),"w")
string = (str(median_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


channels = 'PEDOT'
#RMS
noisefile = r'250noise_RMS_PEDOT.npy'
RMS_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pedot = np.load(RMS_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline_PEDOT.txt"),"w")
string = (str(RMS_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_PEDOT.npy'
median_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pedot = np.load(median_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline_PEDOT.txt"),"w")
string = (str(median_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()



#Neuronexus5 2014-11-25 rec 20_32_48------------------------------------------------------------------------------------
folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_25\rec0'

channels = 'Pristine'
#RMS
noisefile = r'250noise_RMS_pristine.npy'
RMS_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pristine = np.load(RMS_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline_pristine.txt"),"w")
string = (str(RMS_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pristine.npy'
median_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pristine = np.load(median_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline_pristine.txt"),"w")
string = (str(median_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


channels = 'PEDOT'
#RMS
noisefile = r'250noise_RMS_PEDOT.npy'
RMS_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pedot = np.load(RMS_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline_PEDOT.txt"),"w")
string = (str(RMS_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_PEDOT.npy'
median_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pedot = np.load(median_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline_PEDOT.txt"),"w")
string = (str(median_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()



#Neuronexus5 2014-11-25 pair1.0 21_27_13--------------------------------------------------------------------------------
folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_25\pair1.0'

channels = 'Pristine'
#RMS
noisefile = r'250noise_RMS_pristine.npy'
RMS_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pristine = np.load(RMS_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline_pristine.txt"),"w")
string = (str(RMS_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pristine.npy'
median_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pristine = np.load(median_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline_pristine.txt"),"w")
string = (str(median_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


channels = 'PEDOT'
#RMS
noisefile = r'250noise_RMS_PEDOT.npy'
RMS_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pedot = np.load(RMS_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_saline_PEDOT.txt"),"w")
string = (str(RMS_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_PEDOT.npy'
median_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pedot = np.load(median_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_saline_PEDOT.txt"),"w")
string = (str(median_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


#Neuronexus5 2014-11-25 pair2.0 22_44_57--------------------------------------------------------------------------------
folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_25\pair2.0'

channels = 'Pristine'
#RMS
noisefile = r'250noise_RMS_pristine.npy'
RMS_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pristine = np.load(RMS_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_pristine.txt"),"w")
string = (str(RMS_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pristine.npy'
median_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pristine = np.load(median_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_pristine.txt"),"w")
string = (str(median_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


channels = 'PEDOT'
#RMS
noisefile = r'250noise_RMS_PEDOT.npy'
RMS_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pedot = np.load(RMS_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_PEDOT.txt"),"w")
string = (str(RMS_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_PEDOT.npy'
median_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pedot = np.load(median_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_PEDOT.txt"),"w")
string = (str(median_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


#Neuronexus5 2014-11-25 pair3.0 23_00_08--------------------------------------------------------------------------------
folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_25\pair3.0'

channels = 'Pristine'
#RMS
noisefile = r'250noise_RMS_pristine.npy'
RMS_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pristine = np.load(RMS_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_pristine.txt"),"w")
string = (str(RMS_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pristine.npy'
median_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pristine = np.load(median_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_pristine.txt"),"w")
string = (str(median_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


channels = 'PEDOT'
#RMS
noisefile = r'250noise_RMS_PEDOT.npy'
RMS_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pedot = np.load(RMS_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_PEDOT.txt"),"w")
string = (str(RMS_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_PEDOT.npy'
median_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pedot = np.load(median_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_PEDOT.txt"),"w")
string = (str(median_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


#Neuronexus5 2014-11-13 14_59_40----------------------------------------------------------------------------------------
folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_13\14_59_40'

channels = 'Pristine'
#RMS
noisefile = r'250noise_RMS_pristine.npy'
RMS_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pristine = np.load(RMS_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_pristine.txt"),"w")
string = (str(RMS_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pristine.npy'
median_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pristine = np.load(median_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_pristine.txt"),"w")
string = (str(median_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


channels = 'PEDOT'
#RMS
noisefile = r'250noise_RMS_PEDOT.npy'
RMS_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pedot = np.load(RMS_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_PEDOT.txt"),"w")
string = (str(RMS_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_PEDOT.npy'
median_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pedot = np.load(median_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_PEDOT.txt"),"w")
string = (str(median_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


#Neuronexus5 2014-11-13 15_35_31----------------------------------------------------------------------------------------
folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_13\15_35_31'

channels = 'Pristine'
#RMS
noisefile = r'250noise_RMS_pristine.npy'
RMS_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pristine = np.load(RMS_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_pristine.txt"),"w")
string = (str(RMS_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pristine.npy'
median_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pristine = np.load(median_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_pristine.txt"),"w")
string = (str(median_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


channels = 'PEDOT'
#RMS
noisefile = r'250noise_RMS_PEDOT.npy'
RMS_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pedot = np.load(RMS_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_PEDOT.txt"),"w")
string = (str(RMS_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_PEDOT.npy'
median_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pedot = np.load(median_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_PEDOT.txt"),"w")
string = (str(median_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


#Neuronexus5 2014-11-13 18_05_50----------------------------------------------------------------------------------------
folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_13\18_05_50'

channels = 'Pristine'
#RMS
noisefile = r'250noise_RMS_pristine.npy'
RMS_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pristine = np.load(RMS_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_pristine.txt"),"w")
string = (str(RMS_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pristine.npy'
median_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pristine = np.load(median_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_pristine.txt"),"w")
string = (str(median_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


channels = 'PEDOT'
#RMS
noisefile = r'250noise_RMS_PEDOT.npy'
RMS_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pedot = np.load(RMS_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_PEDOT.txt"),"w")
string = (str(RMS_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_PEDOT.npy'
median_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pedot = np.load(median_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_PEDOT.txt"),"w")
string = (str(median_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()

#Neuronexus5 2014-11-13 21_05_14----------------------------------------------------------------------------------------
folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_13\21_05_14'

channels = 'Pristine'
#RMS
noisefile = r'250noise_RMS_pristine.npy'
RMS_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pristine = np.load(RMS_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_pristine.txt"),"w")
string = (str(RMS_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_pristine.npy'
median_pristine = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pristine = np.load(median_pristine)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_pristine.txt"),"w")
string = (str(median_pristine.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()


channels = 'PEDOT'
#RMS
noisefile = r'250noise_RMS_PEDOT.npy'
RMS_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
RMS_pedot = np.load(RMS_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "RMS_PEDOT.txt"),"w")
string = (str(RMS_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()
#MEDIAN
noisefile = r'250noise_Median_PEDOT.npy'
median_pedot = os.path.join(folder + '\\'+ channels +'\\'+noisefile)
median_pedot = np.load(median_pedot)
f = open(os.path.join(os.path.join(folder + '\\'+ channels), "Median_PEDOT.txt"),"w")
string = (str(median_pedot.reshape(16,1))).replace('[', ' ')
string = string.replace(']', ' ')
f.write(string)
f.close()





#Spectrum power density ------------------------------------------------------------------------------------------------

#File names-------------------------------------------------------------------------------------------------------------
#Neuronexus 6 recording 2017-02-02  rec3
analysis_folder = r'Z:\j\Joana Neto\Backup_2017_28_06\PCDisk\Paper Impedance\Neuronexus6\Noise\rec3'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2017-02-02T15_49_35.bin')
sampling_freq = 20000
#Neuronexus 6 recording 2017-02-02 saline
analysis_folder = r'E:\Paper Impedance\Neuronexus6\Noise\RMSSaline'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2017-02-02T10_30_29.bin')
sampling_freq = 20000


#Neuronexus 5 recording 2014-11-25  Pair3.0
analysis_folder = r'Z:\j\Joana Neto\Backup_2017_28_06\PCDisk\Paper Impedance\Neuronexus5\NoisePower\Recording_2014_11_25\pair3.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-25T23_00_08.bin')
sampling_freq = 30000
#Neuronexus 5 recording 2014-11-25  saline w juxta
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Saline noise_2015_05_11\RMS saline_probe and juxta'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2015-05-11T12_16_44.bin')
sampling_freq = 30000
#Neuronexus 5 recording 2014-11-25  saline
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Saline noise_2015_05_11\RMS saline_probe'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2015-05-11T11_59_54.bin')
sampling_freq = 30000



#Open Data--------------------------------------------------------------------------------------------------------------
amp_dtype = np.uint16
Probe_y_digitization = 32768
num_ivm_channels = 32
filtered_data_type = np.float64
voltage_step_size = 0.195e-6
scale_uV = 1000000

filenameSPD = analysis_folder
raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_unfiltered_uV = (temp_unfiltered - Probe_y_digitization) * scale_uV * voltage_step_size


#Types of Data----------------------------------------------------------------------------------------------------------

#Raw data
freq = 0
temp_filtered_uV = temp_unfiltered_uV


#Highpass 250Hz
high_pass_freq = 250
freq = 'H'+ str(high_pass_freq)
def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=100.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)

temp_filtered_uV = highpass(temp_unfiltered_uV, F_HIGH = (sampling_freq/2)*0.95, sampleFreq = sampling_freq, passFreq = high_pass_freq)


#Lowpass 300Hz
low_pass_freq = 250
freq = 'L'+ str(low_pass_freq)
iir_params = {'order': 3, 'ftype': 'butter', 'padlen': 0}

raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_unfiltered_uV = (temp_unfiltered - Probe_y_digitization) * scale_uV * voltage_step_size

temp_filtered_uV = filters.low_pass_filter(temp_unfiltered_uV, sampling_freq, low_pass_freq, method='iir', iir_params=iir_params)



#Compute SPD for ALL channels, plot and save ---------------------------------------------------------------------------
plt.figure()
Pxx_dens_g =[]
for i in np.arange(num_ivm_channels):
    Pxx_dens, f = mlab.psd(temp_filtered_uV[i,:], sampling_freq, sampling_freq)
    Pxx_dens_g.append(Pxx_dens)
    plt. semilogy(f, Pxx_dens, color='k', linewidth=1, alpha=0.2)
    plt.semilogx()
    #plt.ylim([1e-5,1000000])
    #plt.xlim([2,17000])
    #plt.ylim([0.5e-3,100])
    #plt.xlim([1,6000])
    plt.xlabel('frequency(Hz)')
    plt.ylabel('PSD (uV^2/Hz)')

Pxx_dens_g_matrix = np.array(Pxx_dens_g)
Pxx_dens_g_tranp = Pxx_dens_g_matrix.T
np.savetxt(filenameSPD + '\\' + 'power_' + str(freq) + 'Hz' + '.txt', Pxx_dens_g_tranp, delimiter=',')
np.savetxt(filenameSPD + '\\' + 'frequency_' + str(freq) + 'Hz' +'.txt', f, delimiter=',')
filename_power = os.path.join(filenameSPD + '\\' + 'power_' + str(freq) + 'Hz' +'.npy')
np.save(filename_power, Pxx_dens_g_tranp)




#Integration of power across frequencies--------------------------------------------------------------------------------
power_channels = []
for i in np.arange(num_ivm_channels):
    integrate_power = np.trapz(Pxx_dens_g_matrix[i][0.1:7500])
    power_channels.append(integrate_power)

power_matrix = np.array(power_channels)
power = np.append(np.arange(num_ivm_channels).reshape(num_ivm_channels,1), power_matrix.reshape(num_ivm_channels,1), 1)

np.savetxt(filenameSPD + '\\' + 'Powerintegration_' + str(freq) + 'Hz' + '.txt', power, delimiter=',')


#Compute SPD for PEDOT and pristine channels, plot and save-------------------------------------------------------------
power_pedot=[]
plt.figure()
for i in channels_pedot:
    Pxx_dens,f = mlab.psd(temp_filtered_uV[i,:], sampling_freq, sampling_freq)
    power_pedot.append(Pxx_dens)
    plt. semilogy(f, Pxx_dens,color='r',linewidth=1, label='PEDOT')
    plt.semilogx()
    #plt.ylim([1e-5,1000000])
    #plt.xlim([2,17000])
    #plt.ylim([0.5e-3,100])
    #plt.xlim([1,6000])

power_pristine=[]
for i in channels_pristine:
    Pxx_dens,f = mlab.psd(temp_filtered_uV[i,:], sampling_freq, sampling_freq)
    power_pristine.append(Pxx_dens)
    plt. semilogy(f, Pxx_dens,color='b',linewidth=1, label='pristine')
    plt.semilogx()
    plt.xlabel('frequency(Hz)')
    plt.ylabel('PSD (uV^2/Hz)')
    #plt.legend()
    #plt.ylim([1e-5,1000000])
    #plt.xlim([2,17000])

power_pedot_matrix = np.array(power_pedot)
power_pedot_T = power_pedot_matrix.T
power_pedot_averg= np.average(power_pedot, axis=0)
np.savetxt(filenameSPD + '\\' + 'power_pedot' + str(freq) + 'Hz' + '.txt', power_pedot_T, delimiter=',')
np.savetxt(filenameSPD + '\\' + 'frequency_pedot' +  str(freq) + 'Hz' +'.txt', f, delimiter=',')
filename_power = os.path.join(filenameSPD + '\\' + 'power_pedot' + str(freq) + 'Hz' +'.npy')
np.save(filename_power, power_pedot_T)

power_pristine_matrix = np.array(power_pristine)
power_pristine_T = power_pristine_matrix.T
power_pristine_averg = np.average(power_pristine, axis=0)
np.savetxt(filenameSPD + '\\' + 'power_pristine' + str(freq) + 'Hz' + '.txt', power_pedot_T, delimiter=',')
np.savetxt(filenameSPD + '\\' + 'frequency_pristine' + str(freq) + 'Hz' +'.txt', f, delimiter=',')
filename_power = os.path.join(filenameSPD + '\\' + 'power_pristine' + str(freq) + 'Hz' +'.npy')
np.save(filename_power, power_pristine_T)


plt.figure()
plt. semilogy(f, power_pedot_averg,color='r',linewidth=1, label='PEDOT')
plt. semilogy(f, power_pristine_averg,color='b',linewidth=1, label ='Pristine')
plt.semilogx()
plt.ylim([1e-5,1000000])
plt.xlim([2,17000])
plt.legend()



























#code not in use

#Plot-------------------------------------------------------------------------------------------------------------------

raw_data_file_ivm = r"Z:\l\Lorenza\Videogame_Assay\AK_5.1\2016_11_05-12_46\amplifier.bin"

raw_data_file_ivm = r"Z:\l\Lorenza\Videogame_Assay\AK_5.1\2016_11_06-12_11\amplifier.bin"

raw_data_file_ivm = r"Z:\l\Lorenza\Videogame_Assay\AK_5.1\2016_11_09-15_30\amplifier.bin"


amp_dtype = np.uint16
num_ivm_channels = 32
sampling_freq = 30000
filtered_data_type = np.float64
voltage_step_size = 0.195e-6
scale_uV = 1000000
Probe_y_digitization = 32768

raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_unfiltered_uV = (temp_unfiltered - Probe_y_digitization) * scale_uV * voltage_step_size
high_pass_freq = 250
#temp_filtered_uV =temp_unfiltered_uV


def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=100.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)


time_samples = 1000000
index1 = np.int(temp_unfiltered_uV.shape[1]/10*2)
index2 = np.int(index1 + time_samples)


temp_filtered_uV = highpass(temp_unfiltered_uV[:, index1:index2], F_HIGH = (sampling_freq/2)*0.95, sampleFreq = sampling_freq, passFreq = high_pass_freq)



#Low pass filter--------------------------------------------------------------------------------------------------------
low_pass_freq = 5000

import scipy.signal as signal
from BrainDataAnalysis._Old_Structures import Constants as ct


def low_pass_filter(data, Fsampling, Fcutoff, filterType='but', filterOrder=None, filterDirection='twopass'):
    """
    Low passes the data at the Fcutoff frequency.
    filterType = ´but´ (butterworth) (default) OR ´fir´
    filterOrder = the order of the filter. For the default butterworth filter it is 6
    filterDirection = FilterDirection which defines whether the filter is passed over the data once (and how) or twice
    """
    Wn = np.float32(Fcutoff / (Fsampling / 2.0))
    if filterType == 'fir':
        if filterOrder == None:
            raise ArithmeticError("A filter order is required if the filter is to be a fir and not a but")
        (b, a) = signal.firwin(filterOrder + 1, Wn, width=None, window='hamming', pass_zero=True, scale=True, nyq=1.0)
    else:
        if filterOrder == None:
            filterOrder = 6
        (b, a) = signal.butter(filterOrder, Wn, btype='lowpass', analog=0, output='ba')

    dims = data.ndim
    axis = 0
    if dims == 2:
        axis = 1

    if filterDirection == ct.FilterDirection.TWO_PASS:
        filteredData = signal.filtfilt(b, a, data, axis)
    elif filterDirection == ct.FilterDirection.ONE_PASS:
        filteredData = signal.lfilter(b, a, data, axis, zi=None)
    elif filterDirection == ct.FilterDirection.ONE_PASS_REVERSE:
        data = np.fliplr(data)
        filteredData = signal.lfilter(b, a, data, axis, zi=None)
        filteredData = np.fliplr(filteredData)
    return filteredData



iir_params = {'order': 3, 'ftype': 'butter', 'padlen': 0}
temp_filtered_uV = low_pass_filter(temp_unfiltered_uV, sampling_freq, low_pass_freq, filterOrder=3)



# Built-in Python libraries
import os # functions for interacting w operating system

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
import matplotlib.pyplot as plt # makes matplotlib work like MATLAB. ’pyplot’ functions.
import matplotlib.cm as cm

samp_spec = range(0,200000)
f, t, Sxx = signal.spectrogram(temp_unfiltered_uV[22,samp_spec], sampling_freq, window='hanning', nperseg=1000, noverlap=1000-1, mode='psd')
fmax = 30

x_mesh, y_mesh = np.meshgrid(t, f[f<fmax])
plt.figure(figsize=(11,4))
plt.subplot(2,1,1)
plt.title('Time-Frequency Profile of Local Field Potential', size=20)
#plt.xlim()
plt.pcolormesh(x_mesh, y_mesh, np.log10(Sxx[f<fmax]), cmap=cm.jet)#, vmin=vmin, vmax=vmax)
plt.ylabel('frequency (Hz)')
plt.colorbar()

plt.subplot(2,1,2)
plt.plot(t_index[samp_spec],lfp_data[samp_spec],'k')
plt.xlabel('time (s)')
plt.ylabel('voltage (a.u.)')




plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(temp_unfiltered_uV[22,:], Fs=sampling_freq)

plot.xlabel('Time')

plot.ylabel('Frequency')

offset_microvolt = 200
plt.figure()
for i in np.arange(0, np.shape(temp_filtered_uV)[0]):
    plt.plot(temp_filtered_uV[i, :].T + i*offset_microvolt, linewidth=0.8)
    #plt.title(np.str((APlfp,index1,index2)))
    plt.show()


#Protocol2 to calculate the stdv from noise RMS
# RMS noise level for pristine vs pedot channels

channels_pedot= [22,29,3,23,18,4,17,19,5,16,20,6,30,21,7,31]

channels_pristine= [2,9,28,13,8,27,12,14,26,11,15,25,10,1,24,0]

def RMS_calculation(data):

    RMS = np.zeros(len(channels_pedot))

    for i in range(len(channels_pedot)):
        RMS[i] = np.sqrt((1/len(data[i]))*np.sum(data[i]**2))

    return RMS

noise_rms_pedot = RMS_calculation(temp_filtered_uV[channels_pedot])
noise_rms_pristine = RMS_calculation(temp_filtered_uV[channels_pristine])

noise_rms_average_pedot = np.average(noise_rms_pedot)
noise_rms_average_pristine = np.average(noise_rms_pristine)

noise_rms_stdv_pedot = stats.sem(noise_rms_pedot)
noise_rms_stdv_pristine = stats.sem(noise_rms_pristine)

print('#------------------------------------------------------')
print('RMSpedot:'+ str(noise_rms_pedot))
print('RMS_averagepedot:'+ str(noise_rms_average_pedot))
print('RMS_average_stdvpedot:'+ str(noise_rms_stdv_pedot))

print('RMSpristine:'+ str(noise_rms_pristine))
print('RMS_averagepridtine:'+ str(noise_rms_average_pristine))
print('RMS_average_stdvpristine:'+ str(noise_rms_stdv_pristine))
print('#------------------------------------------------------')

filename_RMS = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_RMS_PEDOT' + '.npy')
np.save(filename_RMS, noise_rms_pedot)
filename_RMS = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_RMS_Pristine' + '.npy')
np.save(filename_RMS, noise_rms_pristine)

#Protocol1 to calculate the stdv from noise MEDIAN

noise_medianpedot = np.median(np.abs(temp_filtered_uV[channels_pedot])/0.6745, axis=1)
noise_medianpristine = np.median(np.abs(temp_filtered_uV[channels_pristine])/0.6745, axis=1)

noise_median_averagepedot = np.average(noise_medianpedot)
noise_median_averagepristine = np.average(noise_medianpristine)

noise_median_stdvpedot = stats.sem(noise_medianpedot)
noise_median_stdvpristine = stats.sem(noise_medianpristine)


print('#------------------------------------------------------')
print('Noise_MedianPristine:'+ str(noise_medianpristine))
print('Noise_Median_averagePristine:'+ str(noise_median_averagepristine))
print('Noise_Median_stdvPristine:'+ str(noise_median_stdvpristine))
print('Noise_MedianPedot:'+ str(noise_medianpedot))
print('Noise_Median_averagePedot:'+ str(noise_median_averagepedot))
print('Noise_Median_stdvPedot:'+ str(noise_median_stdvpedot))
print('#------------------------------------------------------')

filename_Median = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_Median_Pristine' + '.npy')
np.save(filename_Median, noise_medianpristine)

filename_Median = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_Median_PEDOT' + '.npy')
np.save(filename_Median, noise_medianpedot)


#LFP analysis-----------------------------------------------------------------------------------------------------------

#Filenames--------------------------------------------------------------------------------------------------------------
#Neuronexus 6 recording 2017-02-02  rec3
analysis_folder = r'E:\Paper Impedance\Neuronexus6\Noise\rec3'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2017-02-02T15_49_35.bin')
sampling_freq = 20000

#Neuronexus 5 recording 2014-11-25  Pair3.0
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_25\pair3.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-25T23_00_08.bin')
sampling_freq = 30000



amp_dtype = np.uint16
Probe_y_digitization = 32768
num_ivm_channels = 32
#sampling_freq = 20000
filtered_data_type = np.float64
voltage_step_size = 0.195e-6
scale_uV = 1000000


#LOW-PASS
low_pass_freq = 300
iir_params = {'order': 3, 'ftype': 'butter', 'padlen': 0}

raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_unfiltered_uV = (temp_unfiltered - Probe_y_digitization) * scale_uV * voltage_step_size

temp_filtered_uV = filters.low_pass_filter(temp_unfiltered_uV, sampling_freq, low_pass_freq, method='iir', iir_params=iir_params)

#plot signals 10sec---------------------------------
num_samples=temp_filtered_uV.shape[1]
sample_axis= np.arange(0,num_samples)
time_axis= sample_axis/sampling_freq

channel_to_plot = 22
sec = 10
tf= sampling_freq*sec
samples0= sampling_freq
samples1=sampling_freq + tf

#plt.figure()
plt.plot(time_axis[0:tf], temp_filtered_uV[channel_to_plot, samples0:samples1].T, color='r', label= 'Neuronexus5')
#plt.plot(time_axis[0:tf], temp_filtered_uV[channel_to_plot, samples0:samples1].T, color='r', label= 'Neuronexus6')
#plt.ylim(-50,50)
plt.ylim(-1000,1000)
plt.ylabel('Voltage (\u00B5V)', fontsize=20)
plt.xlabel('Time (s)',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend()



#--------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------



def plot_average_extra(temp_filtered_uV, yoffset=100):
    origin_cmap= plt.get_cmap('hsv')
    shrunk_cmap = shiftedColorMap(origin_cmap, start=0.6, midpoint=0.7, stop=1, name='shrunk')
    cm=shrunk_cmap
    cNorm=colors.Normalize(vmin=0,vmax=32)
    scalarMap= cmx.ScalarMappable(norm=cNorm,cmap=cm)
    sites_order_geometry= [0,31,24,7,1,21,10,30,25,6,15,20,11,16,26,5,14,19,12,17,27,4,8,18,13,23,28,3,9,29,2,22]

    num_samples = temp_filtered_uV.shape[1]
    sample_axis= np.arange(-(num_samples/2),(num_samples/2))
    time_axis= sample_axis/30000
    plt.figure()
    for m in np.arange(np.shape(temp_filtered_uV)[0]):
        colorVal=scalarMap.to_rgba(np.shape(temp_filtered_uV)[0]-m)
        plt.plot(time_axis*scale_ms,temp_filtered_uV[sites_order_geometry[m],:].T, color=colorVal)
        #plt.xlim(-2, 2) #window 4ms



#Raw data w/o filtering (plot, RMS, SPD)

# Calculate ELECTRONIC noise (saline solution, resistors... all the data was saved from uint16 to int16, therefore data is around zero)
num_ivm_channels = 32
amp_dtype = np.int16
sampling_freq = 30000
filtered_data_type = np.float64
voltage_step_size = 0.195e-6
scale_uV = 1000000
channels_pedot= [22,29,3,23,18,4,17,19,5,16,20,6,30,21,7,31]
channels_pristine= [2,9,28,13,8,27,12,14,26,11,15,25,10,1,24,0]

#Open Data-------------------------------------------

#raw_data_file_ivm = r'F:\Materials paper w Pedro Baiao\Saline noise_11_05_2015_PAPER\RMS saline_probe\amplifier2015-05-11T11_59_54_sign162016-09-29T10_23_53.bin'
#raw_data_file_ivm = r'F:\Materials paper w Pedro Baiao\Electronic noise_05_05_2015_ PAPER\9.9MOhm\amplifier2015-05-05T18_36_49_int162016-09-29T15_37_44.bin'
#raw_data_file_ivm = r'F:\Materials paper w Pedro Baiao\Electronic noise_05_05_2015_ PAPER\1kOhm\amplifier2015-05-11T12_38_44_int162016-09-29T16_09_20.bin'
#raw_data_file_ivm = r'F:\Materials paper w Pedro Baiao\Electronic noise_05_05_2015_ PAPER\SC-headstage\amplifier2015-05-05T17_20_09_int162016-09-29T17_46_33.bin' #noise SC
raw_data_file_ivm = r'H:\2016-10-26\lori_32_pedotvspristine2016-10-25T12_14_01.bin'
raw_data_file_ivm = r'H:\2016-10-26\lori_RMS_saline_int162016-11-01T19_59_18.bin'

raw_data_file_ivm =r'F:\DataKilosort\32chprobe\amplifier2014-11-25T23_00_08\amplifier2014-11-25T23_00_08.bin'
raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_filtered_uV = temp_unfiltered * scale_uV * voltage_step_size


#PLOT---------------------------------
num_samples=temp_filtered_uV.shape[1]
sample_axis= np.arange(0,num_samples)
time_axis= sample_axis/30000

plt.figure()#3sec
plt.plot(time_axis[0:90000], temp_filtered_uV[channels_pristine, 30000:120000].T, color='b', label= 'Pristine')
plt.plot(time_axis[0:90000], temp_filtered_uV[channels_pedot, 30000:120000].T, color='r', label = 'PEDOT')
#plt.ylim(-50,50)
plt.ylim(-1000,1000)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure()#4msec
plt.plot(time_axis[0:30000], temp_filtered_uV[channels_pristine, 30000:60000].T, color='b', label= 'Pristine')
plt.plot(time_axis[0:30000], temp_filtered_uV[channels_pedot, 30000:60000].T, color='r', label = 'PEDOT')
plt.ylim(-500,500)
#plt.ylim(-50,50)
plt.xlim(0.008,0.012)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

#plot average across PEDOT and Pristine signals
temp_filtered_uV_average_PEDOT = np.average(temp_filtered_uV[channels_pedot,:], axis=0)
temp_filtered_uV_average_Pristine = np.average(temp_filtered_uV[channels_pristine,:], axis=0)
num_samples=temp_filtered_uV_average_PEDOT.shape[0]
sample_axis= np.arange(0,num_samples)
time_axis= sample_axis/30000

plt.figure()#1sec
plt.plot(time_axis[0:30000], temp_filtered_uV_average_Pristine[30000:60000], color ='b', label= 'Pristine')
plt.plot(time_axis[0:30000], temp_filtered_uV_average_PEDOT[30000:60000], color='r', label = 'PEDOT')
plt.ylabel('Voltage (\u00B5V)', fontsize=20)
plt.xlabel('Time (s)',fontsize=20)
plt.ylim(-50,50)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend()

plt.figure()#4msec
plt.plot(time_axis[0:30000], temp_filtered_uV_average_Pristine[30000:60000], color ='b', label= 'Pristine')
plt.plot(time_axis[0:30000], temp_filtered_uV_average_PEDOT[30000:60000], color='r', label = 'PEDOT')
plt.ylabel('Voltage (\u00B5V)', fontsize=20)
plt.xlabel('Time (s)',fontsize=20)
plt.ylim(-50,50)
plt.xlim(0.008,0.012)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend()

#ALL 32channels
num_samples=temp_filtered_uV.shape[1]
sample_axis= np.arange(0,num_samples)
time_axis= sample_axis/30000

plt.figure()#1sec
plt.plot(time_axis[0:30000], temp_filtered_uV[:, 30000:60000].T, color='b')
plt.ylim(-50,50)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure()#4msec
plt.plot(time_axis[0:30000], temp_filtered_uV[:, 30000:60000].T, color='b')
plt.ylim(-50,50)
plt.xlim(0.008,0.012)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)




#SPD spectrum---------------------------------------------------------

#Pedot and Pristine channels

plt.figure()
for i in np.arange(len(channels_pedot)):
    Pxx_dens,f = mlab.psd(temp_filtered_uV[channels_pedot[i],:], sampling_freq, sampling_freq)
    plt. semilogy(f, Pxx_dens,color='r',linewidth=1)
    plt.semilogx()
    plt.ylim([1e-5,1000000])
    plt.xlim([2,17000])
    #plt.ylim([0.5e-3,100])
    #plt.xlim([1,6000])
    plt.xlabel('frequency(Hz)')
    plt.ylabel('PSD (uV^2/Hz)')
    np.savetxt(r'F:\SPD\Pxx_dens_PEDOT'+ str(channels_pedot[i])+'.txt', Pxx_dens, delimiter=',')
    np.savetxt(r'F:\SPD\f_PEDOT'+ str(channels_pristine[i])+'.txt', f, delimiter=',')


for i in np.arange(len(channels_pristine)):
    Pxx_dens,f = mlab.psd(temp_filtered_uV[channels_pristine[i],:], sampling_freq, sampling_freq)
    plt. semilogy(f, Pxx_dens,color='b',linewidth=1)
    plt.semilogx()
    plt.ylim([1e-5,1000000])
    plt.xlim([2,17000])
    plt.xlabel('frequency(Hz)')
    plt.ylabel('PSD (uV^2/Hz)')
    np.savetxt(r'F:\SPD\Pxx_dens_Pristine'+ str(channels_pristine[i])+'.txt', Pxx_dens, delimiter=',')
    np.savetxt(r'F:\SPD\f_Pristine'+ str(channels_pristine[i])+'.txt', f, delimiter=',')

#32channels
for i in np.arange(num_ivm_channels):
    Pxx_dens,f = mlab.psd(temp_filtered_uV[i,:], sampling_freq, sampling_freq)
    np.savetxt(r'F:\SPD\SC\Pxx_dens_SC_ch'+ str(i)+'.txt', Pxx_dens, delimiter=',')

#------------------
#LOW-PASS
low_pass_freq = 50
iir_params = {'order': 3, 'ftype': 'butter', 'padlen': 0}

raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_filtered_uV = temp_unfiltered * scale_uV * voltage_step_size

temp_filtered_uV= filters.low_pass_filter(temp_filtered_uV, sampling_freq, low_pass_freq, method='iir', iir_params=iir_params)
#----------------------
#SPD spectrum---------------------------------------------------------


#burst minus pause SPD--------------------------------------------

burst = temp_filtered_uV[:, 1.6*30000:1.85*30000]
pausa = temp_filtered_uV[:, 1.9*30000:2.15*30000]
Pxx_dens_pedot_burst,f = mlab.psd(burst[22,:], sampling_freq, sampling_freq)
Pxx_dens_pedot_pause,f = mlab.psd(pausa[22,:], sampling_freq, sampling_freq)

Pxx_dens_pris_burst,f = mlab.psd(burst[2,:], sampling_freq, sampling_freq)
Pxx_dens_pris_pause,f = mlab.psd(pausa[2,:], sampling_freq, sampling_freq)
diff_pedot =  Pxx_dens_pedot_burst-Pxx_dens_pedot_pause

diff_pristine = Pxx_dens_pris_burst-Pxx_dens_pris_pause

plt.figure()
plt.semilogy(f, diff_pristine,color='b',linewidth=1)
plt. semilogy(f, diff_pedot,color='r',linewidth=1)
plt.semilogx()

#resdon

#PSD
plt.figure()
for i in channels:
    Pxx_dens,f = mlab.psd(temp_filtered[i,:], sampling_freq, sampling_freq)

    np.savetxt(r'E:\Paper Impedance\256chNeuroseeker\Noise\Saline\SPD\AllFreq'+ str(i)+'.txt', Pxx_dens, delimiter=',')
    plt. semilogy(f, Pxx_dens, color='k',linewidth=1,alpha=0.2)
    plt.semilogx()
    plt.ylim([1e-5,1000000])
    plt.xlim([2,17000])
    #plt.ylim([0.5e-3,100])
    #plt.xlim([1,6000])
    plt.xlabel('frequency(Hz)')
    plt.ylabel('PSD (uV^2/Hz)')


#PLOT LFP---------------------------------
num_samples=temp_filtered_uV.shape[1]
sample_axis= np.arange(0,num_samples)
time_axis= sample_axis/sampling_freq

time_s1=0
time_s2=time_axis[-1]*0.1
plt.figure()#3sec
plt.plot(temp_filtered_uV[2].T, color='b', label= 'Pristine')
#plt.ylim(-50,50)
plt.ylim(-1000,1000)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)




plt.figure()#1sec
plt.plot(time_axis[0:30000], temp_filtered_uV[channels_pristine, 30000:60000].T, color='b', label= 'Pristine')
plt.plot(time_axis[0:30000], temp_filtered_uV[channels_pedot, 30000:60000].T, color='r', label = 'PEDOT')
plt.ylim(-50,50)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure()#4msec
plt.plot(time_axis[0:30000], temp_filtered_uV[channels_pristine, 30000:60000].T, color='b', label= 'Pristine')
plt.plot(time_axis[0:30000], temp_filtered_uV[channels_pedot, 30000:60000].T, color='r', label = 'PEDOT')
plt.ylim(-50,50)
plt.xlim(0.008,0.012)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

#plot average across PEDOT and Pristine signals
temp_filtered_uV_average_PEDOT = np.average(temp_filtered_uV[channels_pedot,:], axis=0)
temp_filtered_uV_average_Pristine = np.average(temp_filtered_uV[channels_pristine,:], axis=0)
num_samples=temp_filtered_uV_average_PEDOT.shape[0]
sample_axis= np.arange(0,num_samples)
time_axis= sample_axis/30000

plt.figure()#1sec
plt.plot(time_axis[0:30000], temp_filtered_uV_average_Pristine[30000:60000], color ='b', label= 'Pristine')
plt.plot(time_axis[0:30000], temp_filtered_uV_average_PEDOT[30000:60000], color='r', label = 'PEDOT')
plt.ylabel('Voltage (\u00B5V)', fontsize=20)
plt.xlabel('Time (s)',fontsize=20)
plt.ylim(-50,50)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend()

plt.figure()#4msec
plt.plot(time_axis[0:30000], temp_filtered_uV_average_Pristine[30000:60000], color ='b', label= 'Pristine')
plt.plot(time_axis[0:30000], temp_filtered_uV_average_PEDOT[30000:60000], color='r', label = 'PEDOT')
plt.ylabel('Voltage (\u00B5V)', fontsize=20)
plt.xlabel('Time (s)',fontsize=20)
plt.ylim(-50,50)
plt.xlim(0.008,0.012)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend()

#32channels
num_samples=temp_filtered_uV.shape[1]
sample_axis= np.arange(0,num_samples)
time_axis= sample_axis/30000

plt.figure()#3sec
plt.plot(time_axis[0:90000], temp_filtered_uV[:, 30000:120000].T, color='b')
#plt.ylim(-50,50)
plt.ylim(-500,500)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure()#1sec
plt.plot(time_axis[0:30000], temp_filtered_uV[:, 30000:60000].T, color='b')
plt.ylim(-50,50)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure()#4msec
plt.plot(time_axis[0:30000], temp_filtered_uV[:, 30000:60000].T, color='b')
plt.ylim(-50,50)
plt.xlim(0.008,0.012)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


for i in np.arange(len(channels_pristine)):
    plt.plot(time_axis[0:90000], temp_filtered_uV[channels_pristine[i], 30000:120000].T + i*1000, color='b', label= 'Pristine')







