__author__ = 'George'

import sys
sys.path.append('E:\ScienceProjects\PythonProjects\TheMeaningOfBrain\BrainDataAnalysis')
sys.path.append('E:\ScienceProjects\PythonProjects\TheMeaningOfBrain\IO')
sys.path.append('E:\ScienceProjects\PythonProjects\TheMeaningOfBrain\Layouts\Grids')
import lynxio as lio
import pylab as pl
import numpy as np
import pandas as pd
import os
import BrainDataAnalysis.ploting_functions as pf
import BrainDataAnalysis.timelocked_analysis_functions as tf
import BrainDataAnalysis.filters as filters
import scipy.signal as signal
import scipy as sp
import Layouts.Grids.grids as grids

f_sampling = 32556
f_hp_cutoff = 500
f_lp_cutoff = 1000
f_subsample = 2000
f_mua_lp_cuttof = 2000
f_mua_subsample = 4000
ADBitVolts = 0.0000000305

folder = r'G:\Donders_2013_2015\2015_NeuralynxData\2015-07-10_12-43-23_B14R9_64ECoG_32Atlas_AwakeNoStim_HS1Ref1HS3Ref4'
memap_folder = r'E:\ScienceProjects\PythonProjects\TheMeaningOfBrain\ExperimentSpecificCode\B14R9'

ecog_bad_channels = [14, 30, 31, 34, 35, 36, 37, 38, 39, 40, 41, 45, 46, 48, 49, 53, 55, 57, 60, 61, 62, 63]
probe_bad_channels = [13, 14, 15, 16, 22, 25, 29, 30]

data_probe_hp = pl.load(os.path.join(memap_folder,'data_probe_hp.npy'), mmap_mode='r+')
data_ecog_lp_ss = pl.load(os.path.join(memap_folder,'data_ecog_lp_ss.npy'), mmap_mode='r+')
spike_samples = pl.load(os.path.join(memap_folder,'spike_samples.npy'), mmap_mode=None)
spike_samples_clean = pl.load(os.path.join(memap_folder,'spike_samples_clean.npy'), mmap_mode=None)
spike_times_shaftC = pl.load(os.path.join(memap_folder, 'spike_times_shaftC.npy'), mmap_mode=None)
spike_times_shaftA = pl.load(os.path.join(memap_folder, 'spike_times_shaftA.npy'), mmap_mode=None)

data_ecog_clean = pl.load(os.path.join(memap_folder,'data_ecog_clean.npy'), mmap_mode='r+')
data_ecog_mua_shaftC = pl.load(os.path.join(memap_folder,'data_ecog_mua_shaftC.npy'), mmap_mode='r+')

data_ecog_fft_norm_shaftA = pl.load(os.path.join(memap_folder, 'data_ecog_fft_norm_shaftA.npy'), mmap_mode=None)
data_ecog_fft_norm_shaftC = pl.load(os.path.join(memap_folder, 'data_ecog_fft_norm_shaftC.npy'), mmap_mode=None)

phases_all_shaftA = pl.load(os.path.join(memap_folder, 'phases_all_shaftA.npy'), mmap_mode=None)
phases_all_shaftC = pl.load(os.path.join(memap_folder, 'phases_all_shaftC.npy'), mmap_mode=None)

data = pl.load(os.path.join(memap_folder,'B14R9_raw.npy'), mmap_mode='r+')


# ----------Data generation-----------------
data = lio.read_all_csc(folder,  assume_same_fs=False, memmap=True, memmap_folder=memap_folder, save_for_spikedetekt=False, channels_to_save=None, return_sliced_data=False)
pl.save(os.path.join(memap_folder, 'B14R9_raw.npy'), data)

data_ecog = data[:64,:]
data_probe = data[64:,:]


data_probe_hp = pl.memmap(os.path.join(memap_folder,'data_probe_hp.dat'), dtype='int16', mode='w+', shape=pl.shape(data_probe))
for i in pl.arange(0, pl.shape(data_probe)[0]):
    data_probe_hp[i,:] = filters.high_pass_filter(data_probe[i,:], Fsampling=f_sampling, Fcutoff=f_hp_cutoff)
    data_probe_hp.flush()
    print(i)
pl.save(os.path.join(memap_folder, 'data_probe_hp.npy'), data_probe_hp)


shape_data_ss = (pl.shape(data_ecog)[0], pl.shape(data_ecog)[1]/int(f_sampling/f_subsample))
data_ecog_lp_ss = pl.memmap(os.path.join(memap_folder, 'data_ecog_lp_ss.dat'), dtype='int16', mode='w+', shape=shape_data_ss)
for i in pl.arange(0, pl.shape(data_ecog)[0]):
    data_ecog_lp_ss[i,:] = signal.decimate(filters.low_pass_filter(data_ecog[i,:], Fsampling=f_sampling, Fcutoff=f_lp_cutoff), int(f_sampling/f_subsample))
    data_ecog_lp_ss.flush()
    print(i)
pl.save(os.path.join(memap_folder, 'data_ecog_lp_ss.npy'), data_ecog_lp_ss)


spike_samples = tf.spikedetect(data_probe_hp, threshold_multiplier=6.5, bad_channels=probe_bad_channels)
pl.save(os.path.join(memap_folder, 'spike_samples.npy'), spike_samples)


spike_samples_clean = spike_samples
for i in pl.arange(pl.size(spike_samples_clean)-1,-1,-1):
    data = data_probe_hp[:, spike_samples[i]-60:spike_samples[i]+60]
    stdevs = sp.std(data,1)
    if np.max(data) > 3000 or pl.any(stdevs>600):
        spike_samples_clean = pl.delete(spike_samples_clean, i)
    if i%100==0:
        print(i)
spike_samples_clean = pl.delete(spike_samples_clean, 0)
pl.save(os.path.join(memap_folder, 'spike_samples_clean.npy'), spike_samples_clean)

channels = np.empty(0)
for i in pl.arange(0, pl.size(spike_samples_clean)):
    data = np.array(data_probe_hp[:, spike_samples_clean[i]].tolist())
    channels = np.append(channels, np.argmax(data))
    if i%100==0:
        print(i)
channels_spikes_df = pd.DataFrame([(channels, spike_samples_clean)], columns=['Channels', 'Samples'])

spike_times_shaftA = channels_spikes_df.Samples[0][channels_spikes_df.Channels[0]>7][channels_spikes_df.Channels[0]<16]
spike_times_shaftB = channels_spikes_df.Samples[0][channels_spikes_df.Channels[0]>23]
spike_times_shaftD = channels_spikes_df.Samples[0][channels_spikes_df.Channels[0]<8]
spike_times_shaftC = sp.setxor1d(spike_samples_clean, sp.union1d(spike_times_shaftA, sp.union1d(spike_times_shaftB, spike_times_shaftD)))

pl.save(os.path.join(memap_folder, 'spike_times_shaftA.npy'), spike_times_shaftA)
pl.save(os.path.join(memap_folder, 'spike_times_shaftC.npy'), spike_times_shaftC)


# ----------Analysis---------------------
f_ecog = f_sampling/(int(f_sampling/f_subsample))
spike_times_shaftA_ecog = np.array(spike_times_shaftA * f_ecog / f_sampling, dtype='int')
spike_times_shaftC_ecog = np.array(spike_times_shaftC * f_ecog / f_sampling, dtype='int')
data_ecog_lp_ss_clean = np.delete(data_ecog_lp_ss, ecog_bad_channels, axis=0)


# Generate eMUA for each Shaft
time_around_spike = 2
time_points_around_spike = int(time_around_spike * f_sampling)
data_ecog_clean = np.memmap(os.path.join(memap_folder,'data_ecog_clean.dat'), dtype='int16', mode='w+', shape=(np.shape(data_ecog)[0]-np.size(ecog_bad_channels), np.shape(data_ecog)[1]))
nc = 0
for c in np.arange(0, np.shape(data_ecog)[0]):
    print(c)
    if not np.size(np.intersect1d(np.array([c]), ecog_bad_channels))>0:
        print(c)
        data_ecog_clean[nc, :] = data_ecog[c, :]
        nc = nc +1
        data_ecog_clean.flush()
pl.save(os.path.join(memap_folder, 'data_ecog_clean.npy'), data_ecog_clean)

time_around_spike = 2
f_ecog_mua = f_sampling/(int(f_sampling/f_mua_subsample))
time_points_around_spike = int(time_around_spike * f_sampling)
decimated_time_points_around_spike = int(time_around_spike * f_ecog_mua)
data_ecog_mua_shaftC_shape = (np.shape(data_ecog_clean)[0], 2*decimated_time_points_around_spike, np.size(spike_times_shaftC_ecog))
data_ecog_mua_shaftC = pl.memmap(os.path.join(memap_folder,'data_ecog_mua_shaftC.dat'), dtype='int16', mode='w+', shape=data_ecog_mua_shaftC_shape)
for i in np.arange(0, np.size(spike_times_shaftC_ecog)):
    cut_data = data_ecog_clean[:, spike_times_shaftC_ecog[i]-time_points_around_spike:spike_times_shaftC_ecog[i]+time_points_around_spike]
    if np.shape(cut_data)[1] == 2*time_points_around_spike:
        print('Spike '+str(i))
        lp_ss_cut_data = signal.decimate(filters.low_pass_filter(cut_data, Fsampling=f_sampling, Fcutoff=f_mua_lp_cuttof), int(f_sampling/f_mua_subsample), axis=1)
        print('Low passed and subsampled')
        hp_lp_ss_cut_data = filters.high_pass_filter(lp_ss_cut_data, Fsampling=f_sampling, Fcutoff=f_hp_cutoff)
        print('Hih passed')
        data_ecog_mua_shaftC[:, :, i] = np.transpose(np.transpose(hp_lp_ss_cut_data) - np.transpose(np.mean(hp_lp_ss_cut_data[:,0:f_ecog_mua*0.1], axis=1)))
        print('Baselined')
        data_ecog_mua_shaftC[:, :, i] = np.abs(data_ecog_mua_shaftC[:, :, i])
        print('Rectified')
        data_ecog_mua_shaftC.flush()
pl.save(os.path.join(memap_folder, 'data_ecog_mua_shaftC.npy'), data_ecog_mua_shaftC)


# SpikeTriggeredAverage
time_around_spike = 0.2
time_points_around_spike = int(time_around_spike * f_ecog)
data_sta_shaftA = np.zeros((np.shape(data_ecog_lp_ss_clean)[0], 2*time_points_around_spike))
s = 0
for i in np.arange(np.size(spike_times_shaftA)):
    cut_data = data_ecog_lp_ss_clean[:, spike_times_shaftA_ecog[i]-time_points_around_spike:spike_times_shaftA_ecog[i]+time_points_around_spike]
    if np.shape(cut_data)[1] == 2*time_points_around_spike:
        data_sta_shaftA = data_sta_shaftA + cut_data
        s = s +1
data_sta_shaftA = data_sta_shaftA / s

sta_baseline_shaftA = np.transpose(np.transpose(data_sta_shaftA) - np.transpose(np.mean(data_sta_shaftA, axis=1)))
sta_baseline_shaftC = np.transpose(np.transpose(data_sta_shaftC) - np.transpose(np.mean(data_sta_shaftC, axis=1)))
sta_avg_shaftA = np.mean(sta_baseline_shaftA[:, 415:420], axis=1)
sta_avg_shaftC = np.mean(sta_baseline_shaftC[:, 415:420], axis=1)

# Create a vector of the actual frequencies the FFT does given the number of points used
# (based on the asked for frequencies)
freqs_true = np.array([(0,0,0)])
num_of_cycles = 3
freqs = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
for i in np.arange(0, np.size(freqs)):
    num_of_points = 2*int(num_of_cycles*f_ecog/(2*freqs[i]))
    freq_step = f_ecog/(num_of_points)
    rfftfr = np.fft.rfftfreq(num_of_points, 1/f_ecog)
    freqs_true = np.append(freqs_true, [(k, x, num_of_points) for k, x in enumerate(rfftfr) if x>freqs[i]-0.4*freq_step and x<freqs[i]+0.6*freq_step], axis=0)
freqs_true = np.delete(freqs_true, 0, axis=0)


# Generate the phases of the LFPs centered around all the spikes in a shaft
data = spike_times_shaftA_ecog
phases_all = np.zeros((np.shape(data_ecog_lp_ss_clean)[0], np.shape(freqs_true)[0], np.size(data)), dtype='float32')
for f in np.arange(0, np.shape(freqs_true)[0]):
    points_to_use = int(freqs_true[f][2])
    freq_ind = freqs_true[f][0]
    print('Freq '+str(f))
    for s in np.arange(0, np.size(data)):
        data_for_hann = data_ecog_lp_ss_clean[:, data[s]-points_to_use:data[s]+points_to_use]
        if np.shape(data_for_hann)[1] == int(2*points_to_use): #spike is not too close to the end of the data
            data_with_hann = np.multiply(data_for_hann, sp.signal.get_window('hann',Nx=2*points_to_use))
            data_ecog_fft = np.fft.rfft(data_with_hann, axis=1)
            phases_all[:, f, s] = np.angle(data_ecog_fft)[:, freq_ind]
        else:
            phases_all[:, f, s] = np.nan
        if s%100==0:
            print(' spike '+str(s))
pl.save(os.path.join(memap_folder, 'phases_all_shaftA.npy'), phases_all)

phases_spike_avg = np.mean(phases_all, axis=2)
phases_spike_std = np.std(phases_all, axis=2, dtype='float64')
conf_int = 0.66*phases_spike_std/np.sqrt(np.size(spike_times_shaftC_ecog))
fs = [freqs_true[i][1] for i in np.arange(0, np.shape(freqs_true)[0])]


# Calculate the Pairwise (shaft-ECoG channel) Phase Consistency (PPC)
spikes = spike_times_shaftC_ecog
data_ecog_fft_norm = np.zeros((np.size(spikes), np.shape(data_ecog_lp_ss_clean)[0], np.shape(freqs_true)[0]), dtype='complex64') #spikes X channels X freqs
for f in np.arange(0, np.shape(freqs_true)[0]):
    points_to_use = int(freqs_true[f][2])
    freq_ind = freqs_true[f][0]
    print('Freq '+str(f))
    for s in np.arange(0, np.size(spikes)):
        data_for_hann = data_ecog_lp_ss_clean[:, spikes[s]-points_to_use:spikes[s]+points_to_use]
        if np.shape(data_for_hann)[1] == int(2*points_to_use): #spike is not too close to the end of the data
            data_with_hann = np.multiply(data_for_hann, sp.signal.get_window('hann',Nx=2*points_to_use))
            data_ecog_fft = np.fft.rfft(data_with_hann, axis=1)
            data_ecog_fft_norm[s, :, f] = np.divide(data_ecog_fft[:, freq_ind], np.abs(data_ecog_fft[:, freq_ind]))
        else:
            data_ecog_fft_norm[s, :, f] = np.nan
        if s%100==0:
            print(' spike '+str(s))
pl.save(os.path.join(memap_folder, 'data_ecog_fft_norm_shaftA.npy'), data_ecog_fft_norm)

dof = np.sum(~np.isnan(data_ecog_fft_norm), axis=0)
sinSum = np.abs(np.nansum(np.imag(data_ecog_fft_norm),axis=0));
cosSum = np.nansum(np.real(data_ecog_fft_norm), axis=0);
PPC = (np.square(cosSum)+np.square(sinSum) - dof)/(dof*(dof-1));


thetaPPC = PPC[:,2]
alphaPPC = np.mean(PPC[:,4:9],axis=1)
betaPPC = np.mean(PPC[:,9:16],axis=1)
gammalowPPC = np.mean(PPC[:,16:20],axis=1)
gammahighPPC = np.mean(PPC[:,20:],axis=1)

#----------Visualization----------------
times = spike_times_shaftA
data = data_probe_hp
ploted_points = 64
f_sample = f_sampling
bad_spikes = pl.empty(0)
fig = pl.figure(0)
ax = fig.add_subplot(111)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
def on_pick(event):
    event.artist.set_visible(not event.artist.get_visible())
    print(ax.lines.index(event.artist))
    fig.canvas.draw()
fig.canvas.callbacks.connect('pick_event', on_pick)
lines = ax.plot(pl.arange(-ploted_points/f_sample, ploted_points/f_sample, 1/f_sample), pl.transpose(data[:, times[0]-ploted_points:times[0]+ploted_points]), picker=True)
trial_text = pl.figtext(0.85, 0.85, "Spike: "+str(0), ha="right", va="top", size=20, bbox=bbox_props)
if pl.size(pl.ginput(n=200, mouse_add=1, mouse_stop=3, mouse_pop=2))>0:
        bad_spikes = pl.append(bad_spikes, 0)
        print(0)
for i in pl.arange(pl.size(times)-50, pl.size(times), 5):
    new_data = pl.transpose(data[:, times[i]-ploted_points:times[i]+ploted_points])
    for k in pl.arange(0,pl.size(lines)):
        lines[k].set_ydata(new_data[:, k])
    trial_text.set_text("Spike: "+str(i))
    if pl.size(pl.ginput(n=200, mouse_add=1, mouse_stop=3, mouse_pop=2))>0:
        bad_spikes = pl.append(bad_spikes, i)
        print(i)


plotPPC = gammalowPPC
for i in np.arange(0, np.size(thetaPPC)):
    if np.isnan(gammalowPPC[i]):
        plotPPC[i] = 0

channel_positions = grids.grid_layout_78channels_eric_inverse(bad_channels=np.array(ecog_bad_channels)+1, top_14_channels_removed=True)
pf.plot_topoplot(channel_positions, plotPPC, show=True, zlimits=[-0.01, 0.01])