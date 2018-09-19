
import timelocked_analysis_functions as tl
import ploting_functions as pf
import Utilities as ut
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = "C:\\Users\\IntelligentSystem\kampff.lab@gmail.com\george\Code\Python\ExtraRequirements\\ffmpeg-20140618-git-7f52960-win64-static\\bin\\ffmpeg.exe"
import mne as mne


sampling_freq = 30000


#Cell 1
fn_adc_c1 = "C:\\George\\Data\\ECoG_Juxta_Paired_Recordings\\2014_12_05_Jpak77\\2014_12_05_Jpak77_JuxtaCell1_ADC_Cell1.bin"
fn_amp_c1 = "C:\\George\\Data\\ECoG_Juxta_Paired_Recordings\\2014_12_05_Jpak77\\2014_12_05_Jpak77_JuxtaCell1_Amp_Cell1.bin"

threshold = 0.0003
cell_1_data_raw_spikes = tl.load_raw_event_trace(fn_adc_c1, number_of_channels=8, channel_used=0, dtype=np.uint16)
cell_1_event_times, cell_1_eventdata_in_V = tl.create_spike_triggered_events(cell_1_data_raw_spikes.dataMatrix, threshold = threshold, inter_spike_time_distance=0.001, amp_gain=1000,
                                  sampling_freq=sampling_freq, amp_y_digitization=65536, amp_y_range=10)

cell_1_event_times = cell_1_event_times[1:-120]
cell_1_data_raw_amp = tl.load_raw_data(fn_amp_c1, 64)


cell_1_tl_sub_reref_avg, sub_time_axis = tl.time_lock_raw_data(cell_1_data_raw_amp.dataMatrix, cell_1_event_times, times_to_cut=[-0.2, 0.2], sampling_freq=sampling_freq,
                                                               baseline_time=[-0.2, -0.05], sub_sample_freq=2000, avg_reref = True, keep_trials=False)


x_of_bad_channels = np.array([19, 136, 214, 87, 516])  #For 420 spikes average
y_of_bad_channels = np.array([88, 68, 94, -119, 64])
cell_1_tl_sub_reref_avg_nbc, bad_channels_cell_1_tl_sub_reref_avg = bad_channel_removal(x_of_bad_channels, y_of_bad_channels, cell_1_tl_sub_reref_avg)
pf.plot_video_topoplot(cell_1_tl_sub_reref_avg_nbc, sub_time_axis, pf.grid_layout_64channels(bad_channels_cell_1_tl_sub_reref_avg), times_to_plot = [-0.06, 0.1],
                       time_step = 0.001, sampling_freq = 1000, zlimits = [-60, 60], filename = 'C:\George\Analysis\Jpak77_Juxta\\Cell_1_LFPs.avi')


cell_1_tl_reref_avg, time_axis = tl.time_lock_raw_data(cell_1_data_raw_amp.dataMatrix, cell_1_event_times, times_to_cut=[-0.2, 0.2], sampling_freq=sampling_freq,
                                                       baseline_time=[-0.15, -0.05], sub_sample_freq=None, high_pass_cutoff = None, avg_reref = True, keep_trials=False)


cell_1_tl_avg, time_axis = tl.time_lock_raw_data(cell_1_data_raw_amp.dataMatrix, cell_1_event_times, times_to_cut=[-0.2, 0.2], sampling_freq=sampling_freq,
                                                       baseline_time=[-0.15, -0.05], sub_sample_freq=None, high_pass_cutoff = None, avg_reref = False, keep_trials=False)
x_of_bad_channels = np.array([6500, 1200, 7825, 8720])
y_of_bad_channels = np.array([-70, 44, 63, -54])
cell_1_tl_sub_avg_nbc, bad_channels_cell_1_tl_reref_avg = bad_channel_removal(x_of_bad_channels, y_of_bad_channels, cell_1_tl_sub_avg)




cell_1_tl_hp_avg, hp_time_axis =  tl.time_lock_raw_data(cell_1_data_raw_amp.dataMatrix, cell_1_event_times, times_to_cut=[-0.2, 0.2], sampling_freq=sampling_freq,
                                                           baseline_time=[-0.15, -0.05], sub_sample_freq=None, high_pass_cutoff = 400, avg_reref = False, keep_trials=False)
cell_1_tl_hp_avg = cell_1_tl_hp_avg[:, 500:-500]
hp_time_axis = hp_time_axis[500:-500]
pf.plot_video_topoplot(cell_1_tl_hp_avg, hp_time_axis, pf.grid_layout_64channels(), times_to_plot = [-0.06, 0.06], time_step = 0.0005,
                       sampling_freq = sampling_freq, zlimits = [-0.04, 0.04], filename = 'C:\George\Analysis\Jpak77_Juxta\\Cell_1_HighPassed.avi')

cell_1_tl_reref_hp_avg, hp_time_axis =  tl.time_lock_raw_data(cell_1_data_raw_amp.dataMatrix, cell_1_event_times, times_to_cut=[-0.2, 0.2], sampling_freq=sampling_freq,
                                                           baseline_time=[-0.15, -0.05], sub_sample_freq=None, high_pass_cutoff = 400, avg_reref = True, keep_trials=False)
cell_1_tl_reref_hp_avg = cell_1_tl_reref_hp_avg[:,500:-500]
hp_time_axis = hp_time_axis[500:-500]
pf.plot_video_topoplot(cell_1_tl_reref_hp_avg, hp_time_axis, pf.grid_layout_64channels(), times_to_plot = [-0.06, 0.06], time_step = 0.0005,
                       sampling_freq = sampling_freq, zlimits = [-0.002, 0.002], filename = 'C:\George\Analysis\Jpak77_Juxta\\Cell_1_Reref_HighPassed.avi')



cell_1_tl_sub_reref_allTrials, cell_1_tl_sub_reref_avg, sub_time_axis = tl.time_lock_raw_data(cell_1_data_raw_amp.dataMatrix, cell_1_event_times, times_to_cut=[-0.1, 0.1], sampling_freq=sampling_freq,
                                                               baseline_time=[-0.1, -0.05], sub_sample_freq=2000, avg_reref = True, keep_trials=True)
cell_1_tl_sub_spikes = tl.time_lock_raw_data(cell_1_eventdata_in_V, cell_1_event_times, times_to_cut=[-0.1, 0.1], sampling_freq=sampling_freq,
                                                               baseline_time=[-0.1, -0.05], sub_sample_freq=2000, avg_reref = True, keep_trials=False)

num_of_chunks = int(np.size(cell_1_tl_sub_reref_allTrials,2)/300)
cell_1_tl_sub_reref_chunkedAvg = np.zeros((np.size(cell_1_tl_sub_reref_allTrials, 0), np.size(cell_1_tl_sub_reref_allTrials, 1), num_of_chunks))
temp = np.zeros((np.size(cell_1_tl_sub_reref_allTrials, 0), np.size(cell_1_tl_sub_reref_allTrials, 1)))
for i in np.arange(0, num_of_chunks):
    for k in np.arange(i*300, i*300 + 300):
        temp += cell_1_tl_sub_reref_allTrials[:, :, k]
    cell_1_tl_sub_reref_chunkedAvg[:, :, i] = temp / 300
del temp




#Cell 2
fn_adc_c2 = "C:\\George\\Data\\ECoG_Juxta_Paired_Recordings\\2014_12_05_Jpak77\\2014_12_05_Jpak77_JuxtaCell1_ADC_Cell2.bin"
fn_amp_c2 = "C:\\George\\Data\\ECoG_Juxta_Paired_Recordings\\2014_12_05_Jpak77\\2014_12_05_Jpak77_JuxtaCell1_Amp_Cell2.bin"

threshold = 0.0001
cell_2_data_raw_spikes = tl.load_raw_event_trace(fn_adc_c2, number_of_channels=8, channel_used=0, dtype=np.uint16)
cell_2_event_times, cell_2_eventdata_in_V = tl.create_spike_triggered_events(cell_2_data_raw_spikes.dataMatrix, threshold = threshold, inter_spike_time_distance=0.001, amp_gain=1000,
                                  sampling_freq=sampling_freq, amp_y_digitization=65536, amp_y_range=10)

cell_2_data_raw_amp = tl.load_raw_data(fn_amp_c2, 64)


cell_2_tl_sub_reref_avg, sub_time_axis = tl.time_lock_raw_data(cell_2_data_raw_amp.dataMatrix, cell_2_event_times, times_to_cut=[-0.2, 0.2], sampling_freq=sampling_freq,
                                                               baseline_time=[-0.2, -0.05], sub_sample_freq=2000, avg_reref = True, keep_trials=False)
x_of_bad_channels = np.array([92, 387, 192, 643, 79, 460])
y_of_bad_channels = np.array([73, 48, -51, 32, 34, 27])
cell_2_tl_sub_reref_avg_nbc, bad_channels_cell_1_tl_sub_reref_avg = bad_channel_removal(x_of_bad_channels, y_of_bad_channels, cell_2_tl_sub_reref_avg)


cell_2_tl_hp_avg, hp_time_axis =  tl.time_lock_raw_data(cell_2_data_raw_amp.dataMatrix, cell_2_event_times, times_to_cut=[-0.2, 0.2], sampling_freq=sampling_freq,
                                                           baseline_time=[-0.15, -0.05], sub_sample_freq=None, high_pass_cutoff = 400, avg_reref = False, keep_trials=False)

cell_2_tl_reref_hp_avg, hp_time_axis =  tl.time_lock_raw_data(cell_2_data_raw_amp.dataMatrix, cell_2_event_times, times_to_cut=[-0.2, 0.2], sampling_freq=sampling_freq,
                                                           baseline_time=[-0.15, -0.05], sub_sample_freq=None, high_pass_cutoff = 400, avg_reref = True, keep_trials=False)
pf.plot_video_topoplot(cell_2_tl_reref_hp_avg, hp_time_axis, pf.grid_layout_64channels(), times_to_plot = [-0.02, 0.02], time_step = 0.0005,
                       sampling_freq = sampling_freq, zlimits = [-0.0013, 0.0013], filename = 'C:\George\Analysis\Jpak77_Juxta\\Cell_2_Reref_HighPassed.avi')





#Cell 4
fn_adc_c4 = "C:\\George\\Data\\ECoG_Juxta_Paired_Recordings\\2014_12_05_Jpak77\\2014_12_05_Jpak77_JuxtaCell1_ADC_Cell4.bin"
fn_amp_c4 = "C:\\George\\Data\\ECoG_Juxta_Paired_Recordings\\2014_12_05_Jpak77\\2014_12_05_Jpak77_JuxtaCell1_Amp_Cell4.bin"

threshold = 0.00004
cell_4_data_raw_spikes = tl.load_raw_event_trace(fn_adc_c4, number_of_channels=8, channel_used=0, dtype=np.uint16)
cell_4_event_times, cell_4_eventdata_in_V = tl.create_spike_triggered_events(cell_4_data_raw_spikes.dataMatrix, threshold = threshold, inter_spike_time_distance=0.001, amp_gain=1000,
                                  sampling_freq=sampling_freq, amp_y_digitization=65536, amp_y_range=10)

cell_4_event_times = cell_4_event_times[1:-120] # 'cut the last 2 seconds because np.size(cell_4_eventdata_in_V) - cell_4_event_times[-120] = 676411
cell_4_data_raw_amp = tl.load_raw_data(fn_amp_c4, 64)


cell_4_tl_reref_avg, time_axis = tl.time_lock_raw_data(cell_4_data_raw_amp.dataMatrix, cell_4_event_times, times_to_cut=[-0.2, 0.2], sampling_freq=sampling_freq,
                                                               baseline_time=[-0.2, -0.05], sub_sample_freq=None, avg_reref = True, keep_trials=False)

cell_4_tl_sub_reref_avg, sub_time_axis = tl.time_lock_raw_data(cell_4_data_raw_amp.dataMatrix, cell_4_event_times, times_to_cut=[-0.2, 0.2], sampling_freq=sampling_freq,
                                                               baseline_time=[-0.2, -0.05], sub_sample_freq=2000, avg_reref = True, keep_trials=False)


cell_4_tl_hp_avg, hp4_time_axis =  tl.time_lock_raw_data(cell_4_data_raw_amp.dataMatrix, cell_4_event_times, times_to_cut=[-0.2, 0.2], sampling_freq=sampling_freq,
                                                           baseline_time=[-0.15, -0.05], sub_sample_freq=None, high_pass_cutoff = 400, avg_reref = False, keep_trials=False)
cell_4_tl_hp_avg = cell_4_tl_hp_avg[:, 500:-500]
hp4_time_axis = hp4_time_axis[500:-500]






#Averaging some spikes to see how spike form changes in time
num_of_spikes_in_an_average = 10
num_of_average_waveforms = int(np.size(cell_2_event_times)/num_of_spikes_in_an_average)
points_around_zero = 150

spike_waveforms = np.zeros((2*points_around_zero, num_of_average_waveforms))
temp = np.zeros(2*points_around_zero)

for i in np.arange(0, num_of_average_waveforms):
    for k in np.arange(i*num_of_spikes_in_an_average, i*num_of_spikes_in_an_average + num_of_spikes_in_an_average):
        temp += cell_2_eventdata_in_V[cell_2_event_times[k]-points_around_zero: cell_2_event_times[k]+points_around_zero]

    spike_waveforms[:, i] = temp / num_of_spikes_in_an_average
    temp = np.zeros(2*points_around_zero)

del temp





def bad_channel_removal(x_of_bad_channels, y_of_bad_channels, data):
    x = x_of_bad_channels
    y = y_of_bad_channels
    bad_channels = np.empty(np.size(x), dtype = int)
    for k in np.arange(0, np.size(x)):
        if y[k] > 0:
            temp = [i for i in np.arange(np.shape(data)[0]) if y[k] < 1.1*data[i,int(x[k])] and y[k] > 0.9*data[i,int(x[k])]]
        if y[k] < 0:
            temp = [i for i in np.arange(np.shape(data)[0]) if y[k] > 1.1*data[i,int(x[k])] and y[k] < 0.9*data[i,int(x[k])]]
        if np.size(temp) ==1:
            bad_channels[k] = temp[0]
        else:
            print(x[k])
    data[bad_channels, :] = float('nan')
    #data = np.delete(data, bad_channels, axis=0)
    return data, bad_channels