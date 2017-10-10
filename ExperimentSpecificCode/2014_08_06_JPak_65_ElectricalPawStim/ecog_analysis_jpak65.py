__author__ = 'IntelligentSystem'



import timelocked_analysis_functions as tl
import ploting_functions as pf
import Utilities as ut
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = "C:\\Users\\IntelligentSystem\kampff.lab@gmail.com\george\Code\Python\ExtraRequirements\\ffmpeg-20140618-git-7f52960-win64-static\\bin\\ffmpeg.exe"



fn_adc = "C:\George\Data\ECoG_Chronic_Recordings\\2014_08_06_Jpak65_Anesthesia_PawStimulation_Electrical\\2014_08_06_JPak65_PawStimulation_BackRight_200uA_2ms_ADC.bin"
fn_amp = "C:\George\Data\ECoG_Chronic_Recordings\\2014_08_06_Jpak65_Anesthesia_PawStimulation_Electrical\\2014_08_06_JPak65_PawStimulation_BackRight_200uA_2ms_Amp.bin"


sampling_freq = 20000
threshold = 10000
filt_cutoff_freq = None
minduration = 10
pick_out_or_in = True

data_raw_events = tl.load_raw_event_trace(fn_adc, number_of_channels= 1, dtype=np.uint16)
event_times, event_values = tl.create_piezosensor_events(data_raw_events.dataMatrix, threshold, sampling_freq, filt_cutoff_freq, minduration, pick_out_or_in)

data_raw_amp = tl.load_raw_data(fn_amp, 128)
sub_tl_data, sub_tl_avg_data, sub_time_axis = tl.time_lock_raw_data(data_raw_amp.dataMatrix, event_times, times_to_cut=[-2, 2], sampling_freq=sampling_freq, baseline_time=[-0.5, 0], sub_sample_freq=1000)
#sub_tl_hp_data, sub_tl_hp_avg_data, sub_time_axis = tl.time_lock_raw_data(data_raw_amp.dataMatrix, event_times, times_to_cut=[-2, 2], sampling_freq=sampling_freq, baseline_time=[-0.5, 0], high_pass_cutoff=400, rectify=False, sub_sample_freq=1000)
#sub_tl_hpr_data, sub_tl_hpr_avg_data, sub_time_axis = tl.time_lock_raw_data(data_raw_amp.dataMatrix, event_times, times_to_cut=[-2, 2], sampling_freq=sampling_freq, baseline_time=[-0.5, 0], high_pass_cutoff=400, rectify=True, sub_sample_freq=1000)


fn_adc2 = "C:\George\Data\ECoG_Chronic_Recordings\\14_06_03_Jpak65_Anesthesia_Stimulations\\14_06_03_stimulation_manual_paw_qtip2\stim.bin"
fn_amp2 = "C:\George\Data\ECoG_Chronic_Recordings\\14_06_03_Jpak65_Anesthesia_Stimulations\\14_06_03_stimulation_manual_paw_qtip2\\amplifier.bin"

sampling_freq = 8000
threshold = 1e7
filt_cutoff_freq = 1000
minduration = 100
pick_out_or_in = True
data_raw_events2 = tl.load_raw_event_trace(fn_adc2, number_of_channels=1)
event_times2, event_values2 = tl.create_piezosensor_events(data_raw_events2.dataMatrix, threshold, sampling_freq, filt_cutoff_freq, minduration, pick_out_or_in)

data_raw_amp2 = tl.load_raw_data(fn_amp2, 128)
sub_tl_data2, sub_tl_avg_data2, sub_time_axis2 = tl.time_lock_raw_data(data_raw_amp2.dataMatrix, event_times2, [-1, 1], sampling_freq, baseline_time=[-0.5, 0], sub_sample_freq=1000)





#for spikes
fn_adc = "C:\George\Data\ECoG_Juxta_Paired_Recordings\\14_04_03\Rec2\\adc2014-04-03T19_04_08.bin"
fn_amp = "C:\George\Data\ECoG_Juxta_Paired_Recordings\\14_04_03\Rec2\\amplifier2014-04-03T19_04_08.bin"

sampling_freq = 30000
threshold = 1.2/1000
inter_spike_distance = 0.03
channel = 1
data_raw_spikes = tl.load_raw_event_trace(fn_adc,8, channel)
spike_times, spike_trace_in_V = tl.create_spike_triggered_events(data_raw_spikes.dataMatrix,threshold,inter_spike_distance,1000)

data_raw_ecog = tl.load_raw_data(fn_amp,64,np.float32)
tl_data, tl_avg_trials_data, sub_time_axis = tl.time_lock_raw_data(data_raw_ecog.dataMatrix, spike_times[:-37], times_to_cut=[-2,2], sampling_freq=sampling_freq, baseline_time=[-1.9, -1], sub_sample_freq=1000)
tl_hpr_data, tl__hpr_avg_trials_data, sub_time_axis = tl.time_lock_raw_data(data_raw_ecog.dataMatrix, spike_times[:-37], times_to_cut=[-2,2], sampling_freq=sampling_freq, baseline_time=[-1.9, -1],high_pass_cutoff=400, rectify=True, sub_sample_freq=1000)



k=0
spikes_avg = np.zeros(120000)
spikes_all = np.zeros((120000,np.size(spike_times)))
for i in spike_times:
    if i>60000 and i<np.size(data_raw_spikes)-60000:
        temp = data_raw_spikes[i-60000: i+60000]
        spikes_avg = spikes_avg + temp
        spikes_all[:,k] = temp
        k = k+1
        #plt.plot(np.arange(-2, 2, 1/30000),temp); plt.show()
spikes_avg = spikes_avg/k




sub_tl_avg_trials_data_128 = np.concatenate([sub_tl_avg_trials_data_rostral,sub_tl_avg_trials_data_caudal])
sub_tl_avg_trials_data_128 = np.concatenate([sub_tl_avg_trials_data_rostral,np.zeros([64,4000])])

samples_to_average = np.arange(1924,1980)
#samples_to_average = np.arange(2010,2025)
data_to_plot_rostral = np.mean(sub_tl_avg_trials_data_rostral[:,samples_to_average],1)
data_to_plot_caudal = np.mean(sub_tl_avg_trials_data_caudal[:,samples_to_average],1)
data_to_plot_128 = np.mean(sub_tl_avg_trials_data_128[:,samples_to_average],1)






channel_positions = pf.grid_layout_128channels()
fig = plt.figure()
data = contra_nF_lpF_cl_nF_tl_avg
time_axis = sub_time_axis
times_to_plot = [-0.1,0.2]
time_window = 0.002
time_step = 0.002
zlimits = [-3000, 3000]
sub_sampling_freq = 1000
sub_sampling_offset = sub_sampling_freq*time_axis[0]
sample_step = time_step * sub_sampling_freq
sub_time_indices = np.arange(ut.find_closest(time_axis, times_to_plot[0]), ut.find_closest(time_axis, times_to_plot[1]))
sub_time_indices = sub_time_indices[0::sample_step]
images = []
for t in sub_time_indices:
    samples = [t, t + (time_window*sub_sampling_freq)]
    data_to_plot = np.mean(data[:,samples[0]:samples[1]],1)
    print([((x + sub_sampling_offset) / sub_sampling_freq) for x in samples])
    image, scat = pf.plot_topoplot(channel_positions, data_to_plot, show=False, interpmethod="bicubic", gridscale=5, zlimits = zlimits)
    txt = plt.text(x=2, y=17.5, s=str(time_axis[t])+' secs')
    images.append([image, scat, txt])
FFwriter = animation.FFMpegWriter()
ani = animation.ArtistAnimation(fig, images, interval=1000, blit=True, repeat_delay=1000)
plt.colorbar(mappable=image)
ani.save('C:\George\\Analysis\\JPak65\\14_08_06_Jpak65_Anesthesia_ElectricPawStimulation\\contra_nF_lpF_cl_nF_tl_avg.mp4', writer = FFwriter, fps=1, bitrate=5000, dpi=300, extra_args=['h264'])
plt.show()





store['data'] = pd.DataFrame(data_raw_amp.dataMatrix, index = pd.date_range(start='2014-09-25 11:00:00', periods=np.shape(data_raw_amp.dataMatrix)[1], freq='33333N'), columns = columns)



