

from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
import statsmodels.api as sm


import one_shot_viewer as osv
import sequence_viewer as sv
import transform as tr
import slider as sl

from statsmodels.formula.api import ols

from BehaviorAnalysis import dlc_post_processing as dlc_pp

#  -------------------------------------------------
#  GET FOLDERS
dlc_folder = r'D:\Data\George\AK_33.1\2018_04_30-11_38\Analysis\Deeplabcut'
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-05-07')

markers_file = join(dlc_project_folder, 'videos', r'Croped_videoDeepCut_resnet50_V1May7shuffle1_150000.h5')

labeled_video_file = join(dlc_project_folder, 'videos', r'Croped_videoDeepCut_resnet50_V1May7shuffle1_150000_labeled.mp4')

crop_window_position_file = join(dlc_folder, 'BonsaiCroping', 'Croped_Video_Coords.csv')

#  -------------------------------------------------


# --------------------------------------------------
# --------------------------------------------------
# DEVELOPING THE POST PROCESSING CLEANING OF THE DLC RESULTS
# --------------------------------------------------
# --------------------------------------------------


#  -------------------------------------------------
#  LOAD MARKERS FROM DLC AND TRANSFORM THEIR X, Y COORDINATES BACK TO THE WHOLE ARENA
markers_croped = pd.read_hdf(markers_file)
crop_window_position = pd.read_csv(crop_window_position_file, sep=' ', names=['X', 'Y'], usecols=[4, 5])

markers = dlc_pp.assign_croped_markers_to_full_arena(markers_croped, crop_window_position)
#  -------------------------------------------------


#  -------------------------------------------------
#  HAVE A LOOK AT HOW CLEAN THE BODY MARKERS ARE

frame = 3
sv.image_sequence(globals(), 'frame', 'labeled_video_file')

#  Get the main body parts and nan the low likelihood frames
body_parts = ['Neck', 'Mid Body', 'Tail Base']
body_markers = dlc_pp.seperate_markers(markers, body_parts)


likelihood_threshold = 0.8
body_markers = dlc_pp.turn_low_likelihood_to_nans(body_markers, likelihood_threshold)


'''
body_markers_pos_only = body_markers.loc[:, body_markers.columns.get_level_values(2).isin(['x', 'y'])]
body_markers_pos_only.loc[0, :] = body_markers_pos_only.loc[1, :]
'''

# Average the body markers to a single position
body_positions = dlc_pp.average_multiple_markers_to_single_one(body_markers, flip=True)
body_positions[0, :] = body_positions[1, :]

body_positions = dlc_pp.clean_large_movements(body_positions, maximum_pixels=60)

frames_to_plot = -1
# plt.plot(body_positions[:frames_to_plot, 1], body_positions[:frames_to_plot, 0])

'''
body_velocities = np.diff(body_positions, axis=0)
body_speeds = np.sqrt(np.power(body_velocities[:, 0], 2) + np.power(body_velocities[:, 1], 2))
plt.plot(body_speeds[:frames_to_plot])
'''

global traj_x
traj_x = 0
global traj_y
traj_y = 0


def update_trajectory(f):
    global traj_x
    global traj_y
    traj_x = body_positions[:f, 0]
    traj_y = body_positions[:f, 1]
    return body_positions[:f, :]

traj = None
tr.connect_repl_var(globals(), 'frame', 'update_trajectory', 'traj')


osv.graph(globals(), 'traj_y', 'traj_x')
#  -------------------------------------------------

'''
#  -------------------------------------------------
# FITTING THE MARKERS TO GET BETTER ESTIMATES OF THE LOW LIKELIHOOD ONES
# Fitting 2d surface using multiple markers
# DID NOT WORK

body_markers_positions = markers.loc[:, markers.columns.get_level_values(1).isin(body_parts)]
body_markers_positions = body_markers.loc[:, body_markers.columns.get_level_values(2).isin(['x', 'y'])]


t = np.reshape(body_markers_positions.loc[:3605*120-1, :].values, (3605, 120, 6))
sec = 0
im_lev = [0, 50]
cm = 'jet'
sv.graph_pane(globals(), 'sec', 't')

t1 = np.swapaxes(t, 1, 2)
sv.graph_pane(globals(), 'sec', 't1')


s = np.linspace(0, 2, 3)
npoints = 120
start = 10000
t = np.linspace(start+1, start+npoints, npoints)
S, T = np.meshgrid(s, t)
Z_x = body_markers_positions.loc[start+1:start+npoints, body_markers_positions.columns.get_level_values(2)=='x'].values
Z_y = body_markers_positions.loc[start+1:start+npoints, body_markers_positions.columns.get_level_values(2)=='y'].values

fig_x = plt.figure(1)
ax_x = plt.axes(projection='3d')
ax_x.plot_surface(S, T, Z_x)
fig_y = plt.figure(2)
ax_y = plt.axes(projection='3d')
ax_y.plot_surface(S, T, Z_y)


S_f = S.flatten()
T_f = T.flatten()
Z_x_f = Z_x.flatten()
Z_y_f = Z_y.flatten()
d_x = pd.DataFrame({'s': S_f, 't': T_f, 'z': Z_x_f})
d_y = pd.DataFrame({'s': S_f, 't': T_f, 'z': Z_y_f})

degree = 10
formula = 'I(t**2) + I(s**2) + s*t + s + t'
for i in range(3, degree + 1, 1):
    formula = 'I(t**{}) + '.format(str(i)) + formula
    formula = 'I(s**{}) + '.format(str(i)) + formula
formula = 'z ~ ' + formula
print(formula)

model_x = ols(formula=formula, data=d_x)
model_y = ols(formula=formula, data=d_y)
res_x = model_x.fit()
res_y = model_y.fit()

pred_x = res_x.params['Intercept'] +\
         res_x.params['t'] * T +\
         res_x.params['s:t'] * S * T + res_x.params['s'] * S +\
         res_x.params['I(s ** 2)'] * np.power(S, 2) + \
         res_x.params['I(t ** 2)'] * np.power(T, 2)
for i in range(3, degree + 1, 1):
    f = 'I(t ** {})'.format(str(i))
    pred_x += res_x.params[f] * np.power(T, i)
    f = 'I(s ** {})'.format(str(i))
    pred_x += res_x.params[f] * np.power(S, i)


ax_x.plot_surface(S, T, pred_x)

pred_y = res_y.params[1] * np.power(X, 3) + res_y.params[2] * np.power(Y, 3) + res_y.params[3] * np.power(X, 2) + res_y.params[4] * np.power(Y, 2) + res_y.params['x:y'] * X * Y + res_y.params['x'] * X + res_y.params['y'] * Y + res_y.params['Intercept']
ax_y.plot_surface(X, Y, pred_y)
'''

#  -------------------------------------------------
#  -------------------------------------------------
# CLEANING THE BODY MARKERS BY FITTING 1D LINE USING ONLY ONE MARKER

# Have a look at what different gaps and orders do to the fits
body_markers_positions = body_markers.loc[:, body_markers.columns.get_level_values(2).isin(['x', 'y'])]

column = 0
positions = body_markers_positions.loc[:, body_markers_positions.columns[column]]
if np.isnan(positions[0]):
    positions.loc[0] = positions.loc[1]

gap = 5
order = 3
windows = dlc_pp.find_windows_with_nans(positions, gap=gap)
figure = plt.figure(0)
args = [windows, figure, positions, gap, order]
slider_limits = [0, len(windows)]
window_index = 0
output = None
transform_to_interpolate = dlc_pp.transform_to_interpolate
sl.connect_repl_var(globals(), 'window_index', 'transform_to_interpolate', 'output', 'args', slider_limits)


# Clean all nans
'''
gap = 10
order = 4
updated_markers_filename = join(dlc_project_folder, 'post_processing', 'cleaned_body_marker_positions_order_{}_gap_{}.df'.
                                format(str(order), str(gap)))
updated_body_markers_positions = dlc_pp.clean_dlc_outpout(updated_markers_filename, markers, gap, order)
'''

# Best results for gap = 10 and order = 4
updated_markers_filename = join(dlc_project_folder, 'post_processing', 'cleaned_body_marker_positions_order_{}_gap_{}.df'.
                                format(str(4), str(10)))
updated_body_markers_positions = pd.read_pickle(updated_markers_filename)

# Have a look
body_positions = dlc_pp.average_multiple_markers_to_single_one(updated_body_markers_positions, True)

body_positions = dlc_pp.clean_large_movements(body_positions, maximum_pixels=40)
# --------------------------------------------------


# --------------------------------------------------
# --------------------------------------------------
# DEVELOPING THE VELOCITY CORRELATIONS
# --------------------------------------------------
# --------------------------------------------------

# Load the clean body markers
updated_markers_filename = join(dlc_project_folder, 'post_processing', 'cleaned_body_marker_positions_order_{}_gap_{}.df'.
                                format(str(4), str(10)))
updated_body_markers_positions = pd.read_pickle(updated_markers_filename)

body_positions = dlc_pp.average_multiple_markers_to_single_one(updated_body_markers_positions, flip=True)

body_positions = dlc_pp.clean_large_movements(body_positions, maximum_pixels=20)

# Find speeds
body_velocities = np.diff(body_positions, axis=0)
body_velocities_polar = np.array([np.sqrt(np.power(body_velocities[:, 0], 2) + np.power(body_velocities[:, 1], 2)),
                         180 * (1/np.pi) * np.arctan(body_velocities[:, 1] / body_velocities[:, 0])]).transpose()

# Look at the trajectory with an arrow at the end for the current speeds
frames_to_average = int(120 * 0.5 * 0.5)
def update_trajectory(frame, figure):
    figure.clear()
    traj_x = body_positions[:frame, 0]
    traj_y = body_positions[:frame, 1]
    speed_x = np.nanmean(body_velocities[frame-frames_to_average:frame+frames_to_average, 0]) * frames_to_average
    speed_y = np.nanmean(body_velocities[frame-frames_to_average:frame+frames_to_average, 1]) * frames_to_average
    ax = figure.add_subplot(111)
    ax.plot(traj_x, traj_y)
    ax.arrow(traj_x[-1], traj_y[-1], speed_x, speed_y, head_width=0.5, head_length=0.5)

    return speed_x


fig = plt.figure(0)
frame = 2
out = None
args = [fig]
slider_limits = [frames_to_average, len(body_velocities) - frames_to_average]
sl.connect_repl_var(globals(), 'frame', 'update_trajectory', 'out', 'args', slider_limits)



import sklearn.metrics as skmetrics
ami = skmetrics.adjusted_mutual_info_score(body_positions[:10000, 0], body_positions[:10000, 1], 'arithmetic')
