__author__ = 'George Dimitriadis'




import os.path
import numpy as np
import pandas as pd
import IO.ephys as ephys
import BrainDataAnalysis.timelocked_analysis_functions as tlf
import BrainDataAnalysis.ploting_functions as pf
import pylab as pl
import GUIs.DataFrameViewer.gui_data_frame_viewer as dfv
import BehaviorAnalysis.generating_shuttling_trajectories as st
import mne as mne
import pickle
import BrainDataAnalysis.Utilities as ut


path = r"E:\George\DataDamp\Goncalo_ECoG\ECoG\Data\JPAK_75\2014_12_18-15_25"
sync_path = "sync.bin"
data_path = "amplifier.bin"
session_path = r"Analysis\session.hdf5"

tr_key = 'task/trajectories'
paw_events_key = 'task/events/paws'

flpaw = 'front left paw'
trial_paw_event = 'trial of event'
time_paw_event = 'time of event'

name_traj_point = 'name of trajectory point'
trial_traj_point = 'trial of trajectory point'
frame_traj_point = 'frame of trajectory point'
time_traj_point = 'time of trajectory point'
x_traj_point = 'X of trajectory point'
y_traj_point = 'Y of trajectory point'


session = pd.HDFStore(os.path.join(path, session_path))
paws = session[paw_events_key]
fl_paw = paws[[trial_paw_event, time_paw_event, flpaw]][paws[flpaw] != -1]
fl_paw_with_traj = fl_paw[[trial_paw_event, time_paw_event, flpaw]][paws[trial_paw_event] % 2 == 0]

sync = np.squeeze(ephys.load_raw_data(os.path.join(path, sync_path), numchannels=1, dtype=np.uint8).dataMatrix)
sync_diff = np.diff(sync.astype('int8'))
cam_shutter_closing_samples = np.squeeze((sync_diff < -0.9).nonzero())
start_sample = cam_shutter_closing_samples[fl_paw[flpaw].iloc[0]]
session.close()


# single paw cycle events
lfpaw_start_cycle_touch = pd.DataFrame([[fl_paw[trial_paw_event].iloc[0], fl_paw[flpaw].iloc[0], start_sample]], columns=['trial', 'frame', 'sample'])
for i in np.arange(0, fl_paw.shape[0] - 1):
    trial = fl_paw[trial_paw_event].iloc[i]
    next_trial = fl_paw[trial_paw_event].iloc[i+1]
    if trial == next_trial:
        next_frame = fl_paw[flpaw].iloc[i+1]
        next_sample = cam_shutter_closing_samples[next_frame]
        lfpaw_start_cycle_touch.loc[i+1] = [next_trial, next_frame, next_sample]
lfpaw_start_cycle_touch = lfpaw_start_cycle_touch.sort(columns='frame', ascending=True)
events = lfpaw_start_cycle_touch['sample'].tolist()



# full travel events
fl_paw_with_traj_sorted = fl_paw_with_traj.sort(flpaw, ascending=True)
trials = fl_paw_with_traj_sorted[trial_paw_event].unique().tolist()
wrong_foot_trials = [2, 8, 14, 20, 46, 74]
right_footed_trials = [x for x in trials if x not in wrong_foot_trials]
lfpaw_start_trial_touch = pd.DataFrame( columns=['trial', 'frame', 'sample'])
#all_trials_with_traj = [x for x in fl_paw[trial_paw_event].unique() if x % 2 == 0]
for i in np.arange(0, len(right_footed_trials)):
    t = fl_paw[fl_paw[trial_paw_event] == right_footed_trials[i]].iloc[0]
    trial = t[trial_paw_event]
    frame = t[flpaw]
    sample = cam_shutter_closing_samples[frame]
    lfpaw_start_trial_touch.loc[i] = [trial, frame, sample]
events = lfpaw_start_trial_touch['sample'].tolist()
cp = pf.grid_layout_128channels_rl_rr_cl_cr()



data = ephys.load_raw_data(os.path.join(path, data_path), numchannels=128, dtype=np.uint16).dataMatrix


(tl_trials_lfp, tl_data_lfp, tl_time_axis_lfp) = tlf.time_lock_raw_data(data=data, events=events, times_to_cut=[-3.1, 3], sampling_freq=8000, baseline_time=[-2.5, -2.3], sub_sample_freq=1000,
                       high_pass_cutoff=None, rectify=False, low_pass_cutoff=200, avg_reref=False, keep_trials=True)

(tl_trials_mua, tl_data_mua, tl_time_axis_mua) = tlf.time_lock_raw_data(data=data, events=events, times_to_cut=[-3.1, 3], sampling_freq=8000, baseline_time=[1.7, 2], sub_sample_freq=1000,
                       high_pass_cutoff=400, rectify=True, low_pass_cutoff=100, avg_reref=True, keep_trials=True)


pf.plot_avg_time_locked_data(tl_data_lfp[:,4:], tl_time_axis_lfp[4:])
pf.plot_avg_time_locked_data(tl_data_mua, tl_time_axis_mua, timeToPlot=[-3, 3], labels=False, picker=True)


video_file = r"D:\Protocols\Behavior\Shuttling\ECoG\Data\JPAK_75\2014_12_18-15_25\Analysis\Brain\lfp.avi"
pf.plot_video_topoplot(tl_data_lfp, tl_time_axis_lfp, cp, times_to_plot=[-2, 3], time_window=0.02, time_step=0.01, sampling_freq=1000, zlimits=[-3500, 3500], filename=video_file)

video_file = r"D:\Protocols\Behavior\Shuttling\ECoG\Data\JPAK_75\2014_12_18-15_25\Analysis\Brain\mua.avi"
pf.plot_video_topoplot(tl_data_mua, tl_time_axis_mua, cp, times_to_plot=[-2, 3], time_window=0.02, time_step=0.01, sampling_freq=1000, zlimits=[-5, 15], filename=video_file)


fig = pl.figure(1)
ax2 = pf.plot_avg_time_locked_data(tl_data_lfp, tl_time_axis_lfp, timeToPlot=[-2, 2.5], subplot='311', figure=fig)
ax1 = pf.plot_avg_time_locked_data(tl_data_mua, tl_time_axis_mua, timeToPlot=[-2, 2.5], subplot='312', figure=fig)
ax3 = fig.add_subplot(313, sharex=ax1)
tr_frame_sorted = st.tr.sort(frame_traj_point, ascending=True)
fl_paw_with_traj_sorted = st.fl_paw_with_traj_sorted
for trial in trials:
    if trial not in wrong_foot_trials:
        x_times = tr_frame_sorted[time_traj_point][tr_frame_sorted[trial_traj_point] == trial]
        x_times_s = [k.second + k.microsecond/1000000 for k in x_times]
        time_zero = fl_paw_with_traj_sorted[time_paw_event][fl_paw_with_traj_sorted[trial_paw_event] == trial].iloc[0]
        time_zero_s = time_zero.second + time_zero.microsecond/1000000
        x_diff_times = x_times_s - time_zero_s
        x = tr_frame_sorted[x_traj_point][tr_frame_sorted[trial_traj_point] == trial]
        y = tr_frame_sorted[y_traj_point][tr_frame_sorted[trial_traj_point] == trial]
        ax3.plot(x_diff_times ,y, label=str(trial))
ax3.invert_yaxis()
pl.tight_layout()





x = tl_data_lfp[:, 4:]
times = tl_time_axis_lfp[4:]
sfreq = 1000
adaptive = False
low_bias = True
normalization = 'length'
psd_lowfreqs, freqs_low = mne.time_frequency.multitaper_psd(x, sfreq=sfreq, fmin=2, fmax=10, bandwidth=0.5,
                                adaptive=adaptive, low_bias=low_bias, n_jobs=1,
                                normalization=normalization, verbose=None)
psd_hifreqs, freqs_hi = mne.time_frequency.multitaper_psd(x, sfreq=sfreq, fmin=10, fmax=200, bandwidth=5,
                                adaptive=adaptive, low_bias=low_bias, n_jobs=1,
                                normalization=normalization, verbose=None)
pf.plot_avg_time_locked_data(psd_lowfreqs, freqs_low)
pf.plot_avg_time_locked_data(psd_hifreqs, freqs_hi)



#Using mne to do preprocessing
sfreq = 30000
lfp_info = mne.create_info(ch_names=cp.sort_index(0, by='Numbers', ascending=True).Strings.tolist(), sfreq=sfreq, ch_types=['eeg']*np.shape(cp)[0])
raw_memmaped = mne.io.read_raw_from_memmap(os.path.join(path, data_path), lfp_info, dtype=np.uint16, order='time_chan')

#mne.io.Raw.plot(raw=raw_memmaped, duration=2, start=20, n_channels=20, scalings={'eeg': 8000}, remove_dc=True)

id = 1
events_mne = np.c_[np.array(events), np.zeros(len(events), dtype=int), id * np.ones(len(events), dtype=int)]
baseline = (-2.5, -2.3)
event_id = dict(left_paw=id)
epochs = mne.Epochs(raw_memmaped, events_mne, event_id, -3, 3, proj=True, picks=None, baseline=baseline, preload=True, reject=None)
averaged = epochs.average()

power = pickle.load( open(os.path.join(path, "Analysis\\tfr_power.p"), "rb"))

n_cycles = 3
frequencies = np.arange(5, 60, 3)

from mne.time_frequency import tfr_morlet
power, phase_lock = tfr_morlet(epochs, freqs=frequencies, n_cycles=n_cycles, decim=3000, n_jobs=10)



import gui_tfr_viewer
gui_tfr_viewer.TFR_Viewer(power)

box = (0, 0.8, 0, 1.1)
w, h = [.09, .05]

pos = [[ut.normList([x, y], normalizeTo=0.8, vMin=1, vMax=8)[0], ut.normList([x, y], vMin=1, vMax=16)[1], w, h] for [n, s, (x,y)] in cp.sort_index(0, by='Numbers', ascending=True).values]
layout = mne.layouts.Layout(box, pos, cp.sort_index(0, by='Numbers', ascending=True).Strings, cp.sort_index(0, by='Numbers', ascending=True).Numbers, '128ch')

power.plot_topo(picks=None, tmin=-3, tmax=3, fmin=5, fmax=60, vmin=-3e10, vmax=3e10, layout=layout, layout_scale=None)
