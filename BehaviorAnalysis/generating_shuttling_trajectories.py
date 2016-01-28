__author__ = 'George Dimitriadis'



import numpy as np
import pandas as pd
import pylab as pl
from mpldatacursor import datacursor
import GUIs.DataFrameViewer.gui_data_frame_viewer as dfv


tr_key = 'task/trajectories'
paw_events_key = 'task/events/paws'

flpaw = 'front left paw'
trial_paw_event = 'trial of event'
time_paw_event = 'time of event'
frame_paw_event = 'frame of event'

name_traj_point = 'name of trajectory point'
trial_traj_point = 'trial of trajectory point'
frame_traj_point = 'frame of trajectory point'
time_traj_point = 'time of trajectory point'
x_traj_point = 'X of trajectory point'
y_traj_point = 'Y of trajectory point'

path = r"D:\Protocols\Behavior\Shuttling\ECoG\Data\JPAK_75\2014_12_18-15_25\Analysis\session.hdf5"
session = pd.HDFStore(path)

tr = session[tr_key]
paws = session[paw_events_key]

fl_paw = paws[[trial_paw_event, time_paw_event, flpaw]][paws[flpaw] != -1]

fl_paw_with_traj = fl_paw[[trial_paw_event, time_paw_event, flpaw]][paws[trial_paw_event] % 2 == 0]



#fig1 = pl.figure(1)
#fig2 = pl.figure(2)
#ax1 = fig1.add_subplot(111)
#ax2 = fig2.add_subplot(111)
trials = tr[trial_traj_point].unique().tolist()
wrong_foot_trials = [2, 8, 14, 20, 46, 74]
tr_frame_sorted = tr.sort(frame_traj_point, ascending=True)
fl_paw_with_traj_sorted = fl_paw_with_traj.sort(flpaw, ascending=True)
for trial in trials:
    if trial not in wrong_foot_trials:
        x_times = tr_frame_sorted[time_traj_point][tr_frame_sorted[trial_traj_point] == trial]
        x_times_s = [k.second + k.microsecond/1000000 for k in x_times]
        time_zero = fl_paw_with_traj_sorted[time_paw_event][fl_paw_with_traj_sorted[trial_paw_event] == trial].iloc[0]
        time_zero_s = time_zero.second + time_zero.microsecond/1000000
        x_diff_times = x_times_s - time_zero_s
        x = tr_frame_sorted[x_traj_point][tr_frame_sorted[trial_traj_point] == trial]
        y = tr_frame_sorted[y_traj_point][tr_frame_sorted[trial_traj_point] == trial]
#        ax1.plot(x, y, label=str(trial))
#        ax2.plot(x_diff_times, y, label=str(trial))
#ax1.invert_yaxis()
#ax2.invert_yaxis()
#ax2.invert_xaxis()
#datacursor(hover=True)
#pl.show()




paw_touch_points_in_traj_indices = []
for f in fl_paw_with_traj[flpaw]:
    paw_touch_points_in_traj_indices.append(min(tr[tr[frame_traj_point]-f <= 4][tr[frame_traj_point]-f >= -2].index))


paw_cycle_trajectories = pd.DataFrame(columns=np.arange(0, np.shape(fl_paw_with_traj)[0]-1), index=np.arange(0, 30))
paw_cycle_traject_ypos_in_time = pd.DataFrame(columns=np.arange(0, np.shape(fl_paw_with_traj)[0]-1), index=np.arange(0, 30))
for i in np.arange(0, np.shape(fl_paw_with_traj)[0]-1):
    cycle_points = []
    cycle_points_in_time = []
    trial = tr[trial_traj_point].loc[paw_touch_points_in_traj_indices[i]]
    if trial == tr[trial_traj_point].loc[paw_touch_points_in_traj_indices[i+1]]:
        cycle_traj = tr[tr[trial_traj_point] == trial].loc[paw_touch_points_in_traj_indices[i]:paw_touch_points_in_traj_indices[i+1]]
        for k in np.arange(0, cycle_traj.shape[0]):
            cycle_points.append((cycle_traj[x_traj_point].iloc[k], cycle_traj[y_traj_point].iloc[k]))
            cycle_points_in_time.append((cycle_traj[time_traj_point].iloc[k], cycle_traj[y_traj_point].iloc[k]))
        paw_cycle_trajectories[i].iloc[0:np.shape(cycle_points)[0]] = cycle_points
        paw_cycle_traject_ypos_in_time[i].iloc[0:np.shape(cycle_points)[0]] = cycle_points_in_time
paw_cycle_trajectories = paw_cycle_trajectories.dropna(axis=1, how='all')
paw_cycle_trajectories.columns = np.arange(0, paw_cycle_trajectories.shape[1])
paw_cycle_trajectories = paw_cycle_trajectories.dropna(axis=0, how='all')
paw_cycle_traject_ypos_in_time = paw_cycle_traject_ypos_in_time.dropna(axis=1, how='all')
paw_cycle_traject_ypos_in_time.columns = np.arange(0, paw_cycle_traject_ypos_in_time.shape[1])
paw_cycle_traject_ypos_in_time = paw_cycle_traject_ypos_in_time.dropna(axis=0, how='all')


plot_df = paw_cycle_traject_ypos_in_time
#fig = pl.figure(1)
#ax = fig.add_subplot(111)
""":type : matplotlib.axes.Axes"""
for i in np.arange(0, plot_df.shape[1]):
    xy = plot_df[i].dropna().tolist()
    xy = [list(t) for t in zip(*xy)]
    y = xy[1]
    x = [(((x - xy[0][0])).milliseconds)/1000 for x in xy[0]]
    #x = ut.normList(xy[0], normalizeFrom=0)
    if i != 34 and i != 47 and i != 66 and i != 68:
        pass
#        ax.plot(x, y, label=str(i), picker=True)
#ax.invert_yaxis()
#ax.invert_xaxis()
#datacursor(hover=True)




