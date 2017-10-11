
##### Biological noise of pair recordings where PEDOT vs Pristine
channels_pedot= [22,29,3,23,18,4,17,19,5,16,20,6,30,21,7,31]
channels_pristine= [2,9,28,13,8,27,12,14,26,11,15,25,10,1,24,0]
number_ch_PEDOT= len(channels_pedot)
number_ch_pristine= len(channels_pristine)

voltage_step_size = 0.195e-6
scale_uV = 1000000

for i in np.arange(0, len(good_cells)):
    temp_filtered = all_cells_ivm_filtered_data[good_cells[i]]
    number_windows = np.shape(all_cells_ivm_filtered_data[good_cells[i]])[2]
    stdvs_PEDOT = np.zeros([number_windows,number_ch_PEDOT])
    stdvs_pristine = np.zeros([number_windows,number_ch_pristine])
    for w in np.arange(number_windows):
        stdvs_PEDOT  = np.asarray(np.median(np.abs(temp_filtered[channels_pedot])/0.6745, axis=1))
        stdvs_pristine  = np.asarray(np.median(np.abs(temp_filtered[channels_pristine])/0.6745, axis=1))

    stdvs_PEDOT_uV = stdvs_PEDOT * voltage_step_size * scale_uV
    stdvs_pristine_uV = stdvs_pristine * voltage_step_size * scale_uV
    np.save(os.path.join(analysis_folder,'RMS_PEDOT_Cell'+ good_cells[i] + '.npy'), stdvs_PEDOT_uV)
    np.save(os.path.join(analysis_folder,'RMS_pristine_Cell'+ good_cells[i] + '.npy'), stdvs_pristine_uV)


##### Biological noise of ATLAS
num_of_points_in_spike_trig_ivm = 128
num_of_points_for_padding = 2 * num_of_points_in_spike_trig_ivm

inter_spike_time_distance = 30
amp_gain = 1000
num_ivm_channels = 32
#amp_dtype = np.float32
amp_dtype = np.uint16
sampling_freq = 30000
high_pass_freq = 100
filtered_data_type = np.float64

analysis_folder = r'F:\Materials paper w Pedro Baiao\Results_ATLAS\atlas\2013_07_19_passive probes\Surgery day\21.57h_Aligned Site 30'


#recordings
raw_data_file_ivm = r'F:\Materials paper w Pedro Baiao\Results_ATLAS\2013_07_09_passive probes\Surgery day\recordings2.bin'
raw_data_file_ivm = r'F:\Materials paper w Pedro Baiao\Results_ATLAS\2013_07_19_passive probes\Surgery day\21.57h_Aligned Site 30\recordings0.bin'
raw_data_file_ivm = r'F:\Materials paper w Pedro Baiao\Results_ATLAS\2013_07_19_passive probes\Surgery day\21.47h_ Aligned Site 8\recordings0.bin'
raw_data_file_ivm = r'F:\Materials paper w Pedro Baiao\Results_ATLAS\2013_09_19_passive probes\Surgery Day\1600micro.bin'


#saline
raw_data_file_ivm = r'F:\Materials paper w Pedro Baiao\Results_ATLAS\2013_09_19_passive probes\Surgery Day\saline_recordings\recordings1.bin'
raw_data_file_ivm = r'F:\Materials paper w Pedro Baiao\Results_ATLAS\2013_07_09_passive probes\Surgery day\saline\recordings0.bin'
raw_data_file_ivm = r'H:\2016_08_26\atlas1_RMS_saline.bin'


raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=250.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)


window = 0
window_size_secs = 60
filtered_data_type = np.float64
sampling_freq = 30000
high_pass_freq = 250
window_size = int(window_size_secs * sampling_freq)
voltage_step_size = 0.195e-6
scale_uV = 1000000
temp_unfiltered = raw_data_ivm.dataMatrix
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_filtered = highpass(temp_unfiltered)
num_samples=temp_filtered.shape[1]
sample_axis= np.arange(0,num_samples)
time_axis= sample_axis/30000
channels_small= [29,30,7,5,1,26,25,31,6,13]
channels_big= [28,18,2,3,22,27,4,9,21,23,8,10,19,20,11,12]
stdvs_small  = np.asarray(np.median(np.abs(temp_filtered[channels_small])/0.6745, axis=1))
stdvs_big  = np.asarray(np.median(np.abs(temp_filtered[channels_big])/0.6745, axis=1))
stdvs_big_uV= stdvs_big*scale_uV*voltage_step_size
stdvs_small_uV=stdvs_small*scale_uV*voltage_step_size
print(stdvs_small)
print(stdvs_big)
print(stdvs_small_uV)
print(stdvs_big_uV)
np.save(os.path.join(analysis_folder,'RMS_small_ch' + '.npy'), stdvs_small)
np.save(os.path.join(analysis_folder,'RMS_big_ch'+ '.npy'), stdvs_big)



###Find Noise Median
#stdvs  = np.median(np.abs(temp_filtered)/0.6745, axis=1)
stdvs_PEDOT = np.median(np.abs(temp_filtered[channels_pedot])/0.6745, axis=1)
stdvs_pristine = np.median(np.abs(temp_filtered[channels_pristine])/0.6745, axis=1)


voltage_step_size = 0.195e-6
scale_uV = 1000000
#stdvs_uV= stdvs*voltage_step_size*scale_uV
stdvs_PEDOT_uV= stdvs_PEDOT*voltage_step_size*scale_uV
stdvs_pristine_uV= stdvs_pristine*voltage_step_size*scale_uV

#stdv_average = np.average(stdvs_uV)
#stdv_average = np.average(stdvs_uV[np.where(stdvs_uV < 4)])
stdv_PEDOT_average = np.average(stdvs_PEDOT_uV)
stdv_pristine_average = np.average(stdvs_pristine_uV)

#stdv_stdv = stats.sem(stdvs_uV)
#stdv_stdv = stats.sem(stdvs_uV[np.where(stdvs_uV < 4)])
stdv_PEDOT_stdv = stats.sem(stdvs_PEDOT_uV)
stdv_pristine_stdv = stats.sem(stdvs_pristine_uV)

print(stdv_PEDOT_average)
print(stdv_PEDOT_stdv)
print(stdv_pristine_average)
print(stdv_pristine_stdv)



####GOLD NEURONEXUS2 2014_05_20 and 2014_05_30
#Open Data-------------------------------------------

raw_data_file_ivm= r'F:\DataKilosort\32chprobe\amplifier2014-05-20T20_45_05\amplifier2014-05-20T20_45_05.bin'
raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_filtered_uV = temp_unfiltered

def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=250.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)


temp_filtered_uV = highpass(temp_filtered_uV)

gold_all= [22,25,26,27,28,11,12,13]
pristine_all= np.array([2,9,8,14,15,10,1,24,0,29,3,23,18,4,17,19,5,16,20,6,30,21,7,31])

stdvs_gold = np.median(np.abs(temp_filtered_uV[gold_all])/0.6745, axis=1)
stdvs_pristine = np.median(np.abs(temp_filtered_uV[pristine_all])/0.6745, axis=1)
stdv_gold_average = np.average(stdvs_gold)
stdv_pristine_average = np.average(stdvs_pristine)
stdv_gold_stdv = stats.sem(stdvs_gold)
stdv_pristine_stdv = stats.sem(stdvs_pristine)

print('#------------------------------------------------------')
print('RMS_GOLD:'+ str(stdvs_gold))
print('RMS_Pristine:'+ str(stdvs_pristine))
print('RMS_GOLD_average:'+ str(stdv_gold_average))
print('RMS_GOLD_stdv:'+ str(stdv_gold_stdv))
print('RMS_Pristine_average:'+ str(stdv_pristine_average))
print('RMS_Pristine_stdv:' + str(stdv_pristine_stdv))
print('#------------------------------------------------------')


#GOLD plot
num_samples=temp_filtered_uV.shape[1]
sample_axis= np.arange(0,num_samples)
time_axis= sample_axis/30000
plt.figure()#3sec
plt.plot(time_axis[0:90000], temp_filtered_uV[pristine_all, 30000:120000].T, color='b', label= 'Pristine')
plt.plot(time_axis[0:90000], temp_filtered_uV[gold_all, 30000:120000].T, color='r', label = 'PEDOT')
#plt.ylim(-50,50)
plt.ylim(-500,500)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


#NEURONEXUS3
#Open Data-------------------------------------------

raw_data_file_ivm= r'F:\DataKilosort\32chprobe\amplifier2014-07-25T20_50_28\amplifier2014-07-25T20_50_28.bin'

raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_filtered_uV = temp_unfiltered

def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=250.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)


temp_filtered_uV = highpass(temp_filtered_uV)

gold_nano= [15,16]
pristine= np.array([14,30])
pedot= np.array([20,11])

stdvs_gold = np.median(np.abs(temp_filtered_uV[gold_nano])/0.6745, axis=1)
stdvs_pristine = np.median(np.abs(temp_filtered_uV[pedot])/0.6745, axis=1)
stdv_gold_average = np.average(stdvs_gold)
stdv_pristine_average = np.average(stdvs_pristine)
stdv_gold_stdv = stats.sem(stdvs_gold)
stdv_pristine_stdv = stats.sem(stdvs_pristine)

print('#------------------------------------------------------')
print('RMS_GOLD:'+ str(stdvs_gold))
print('RMS_Pedot:'+ str(stdvs_pristine))
print('RMS_GOLD_average:'+ str(stdv_gold_average))
print('RMS_GOLD_stdv:'+ str(stdv_gold_stdv))
print('RMS_Pedot_average:'+ str(stdv_pristine_average))
print('RMS_Pedot_stdv:' + str(stdv_pristine_stdv))
print('#------------------------------------------------------')


#gold nano plot

num_samples=temp_filtered_uV.shape[1]
sample_axis= np.arange(0,num_samples)
time_axis= sample_axis/30000


plt.figure()#3sec
plt.plot(time_axis[0:90000], temp_filtered_uV[pedot, 30000:120000].T, color='b', label= 'Pristine')
plt.plot(time_axis[0:90000], temp_filtered_uV[gold_nano, 30000:120000].T, color='r', label = 'PEDOT')
#plt.ylim(-50,50)
plt.ylim(-500,500)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)




####128ch probe
def openDataTransform(num_channels=128, dtype=np.uint16):
    filename = os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times[good_cells[0]]+'.bin')
    fdata = np.fromfile(filename, dtype)
    numsamples = int(len(fdata) / num_channels)
    dataMatrix = np.reshape(fdata, (numsamples, num_channels))
    data=np.zeros([num_channels,numsamples])
    data = dataMatrix.T
    return data

data.astype('uint16').tofile(r'C:\Users\KAMPFF-LAB_ANALYSIS4\Desktop\2015_09_03_Cell9_0\teste.bin')