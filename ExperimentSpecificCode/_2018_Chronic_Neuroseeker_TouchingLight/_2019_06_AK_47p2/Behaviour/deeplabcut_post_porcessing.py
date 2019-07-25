

from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import one_shot_viewer as osv
import sequence_viewer as sv
import transform as tr
import slider as sl

from BrainDataAnalysis import binning
from BehaviorAnalysis import dlc_post_processing as dlc_pp
from BehaviorAnalysis import dlc_plotting

from plotting_overlays import overlay_dots


import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.events_sync_funcs as sync_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p2 import constants as const

#  -------------------------------------------------
#  GET FOLDERS
date_folder = 8
analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')

dlc_folder = join(analysis_folder, 'Deeplabcut')

dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-06-30')


data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')
kilosort_folder = join(analysis_folder, 'Kilosort')

mutual_information_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis',
                                 'Results', 'MutualInformation')

events_folder = join(data_folder, "events")

markers_file = join(dlc_project_folder, 'videos', r'Croped_videoDeepCut_resnet50_V1Jun30shuffle1_325000.h5')

labeled_video_file = join(dlc_project_folder, 'videos',
                          r'Croped_videoDeepCut_resnet50_V1Jun30shuffle1_325000_labeled.mp4')

crop_window_position_file = join(dlc_folder, 'BonsaiCroping', 'Croped_Video_Coords.csv')

full_video_file = join(dlc_folder, 'BonsaiCroping', 'Full_video.avi')

#  -------------------------------------------------


#  -------------------------------------------------
#  LOAD MARKERS FROM DLC AND TRANSFORM THEIR X, Y COORDINATES BACK TO THE WHOLE ARENA
markers_croped = pd.read_hdf(markers_file)
crop_window_position = pd.read_csv(crop_window_position_file, sep=' ', names=['X', 'Y'], usecols=[4, 5])
crop_window_position.iloc[1:] = crop_window_position.iloc[:-1].values
markers = dlc_pp.assign_croped_markers_to_full_arena(markers_croped, crop_window_position)

#  -------------------------------------------------
# MARKER NAMES OF THE DIFFERENT BODY PARTS

body_parts = ['Neck', 'Body', 'TailBase']
head_parts = ['Nose', 'Ear Left', 'Ear Right', 'Implant_Bottom_Back', 'Implant_Bottom_Front', 'Implant_Top_Back',
              'Implant_Top_Front', 'Implant_Top_Middle']
head_parts_top_vector = ['Implant_Top_Back', 'Implant_Top_Front', 'Implant_Top_Middle']
head_parts_bottom_vector = ['Nose', 'Implant_Bottom_Back', 'Implant_Bottom_Front', 'Neck']
tail_parts = ['TailBase', 'TailMiddle', 'TailTip']
good_parts = ['Implant_Top_Back', 'Implant_Top_Front', 'Implant_Top_Middle',
              'Neck', 'Body', 'TailBase', 'TailMiddle', 'TailTip']


# -------------------------------------------------
# CLEAN
# Run once
# Make all low likelihood positions into nan and remove any sudden, large moves
likelihood_threshold = 0.8
markers_positions = dlc_pp.turn_low_likelihood_to_nans(markers, likelihood_threshold)\
            .loc[:, markers.columns.get_level_values(2).isin(['x', 'y'])]

markers_positions_no_large_movs = dlc_pp.clean_markers_with_large_movements(markers_positions, 80)
pd.to_pickle(markers_positions_no_large_movs, join(dlc_project_folder, 'post_processing',
                                                   'markers_positions_no_large_movs.df'))


# Cleaning the nans away from the markers by extrapolation didn't work. Too many of them
'''
gap = 10
order = 4
updated_markers_filename = join(dlc_project_folder, 'post_processing', 'no_nans_marker_positions_order_{}_gap_{}.df'.
                                format(str(order), str(gap)))
# updated_markers_positions = dlc_pp.clean_dlc_outpout(updated_markers_filename, markers_positions, gap, order)


gap = 2
order = 2
updated_markers_filename = join(dlc_project_folder, 'post_processing', 'no_nans_marker_positions_order_{}_gap_{}.df'.
                                format(str(order), str(gap)))
updated_markers_positions = dlc_pp.clean_dlc_outpout(updated_markers_filename, markers_positions_no_large_movs, gap, order)
'''


# -------------------------------------------------
# HAVE A LOOK AT THE 'CLEANED' DATA

markers_positions_no_large_movs = pd.read_pickle(join(dlc_project_folder, 'post_processing',
                                                 'markers_positions_no_large_movs.df'))

frame = 3
global marker_dots
marker_dots = np.zeros((640, 640, 4))
output = None
overlay_dots.marker_dots = marker_dots
marker_size = 3
markers_positions_no_large_movs_numpy = markers_positions_no_large_movs.values
args = [markers_positions_no_large_movs_numpy, marker_size]
update_markers_for_video = overlay_dots.update_markers_for_video

tr.connect_repl_var(globals(), 'frame', 'output', 'update_markers_for_video', 'args')
sv.image_sequence(globals(), 'frame', 'marker_dots', 'full_video_file')


# -------------------------------------------------
# GET THE BODY POSITIONS

# Get the body markers from the cleaned markers
# Then interpolate the nans of the TailBase
# Finally average the body markers to get the body position
updated_body_markers = dlc_pp.seperate_markers(markers_positions_no_large_movs, body_parts)
tail_base = updated_body_markers.loc[:, updated_body_markers.columns.get_level_values(1) == 'TailBase']
tail_base_cleaned = dlc_pp.clean_dlc_outpout(join(dlc_project_folder, 'post_processing', 'test.df'),
                                             tail_base, gap=1, order=2)
updated_body_markers.loc[:, updated_body_markers.columns.get_level_values(1) == 'TailBase'] = tail_base_cleaned
body_positions = dlc_pp.average_multiple_markers_to_single_one(updated_body_markers, flip=True)
np.save(join(dlc_project_folder, 'post_processing', 'body_positions.npy'), body_positions)

body_positions = np.load(join(dlc_project_folder, 'post_processing', 'body_positions.npy'))

updated_head_markers = dlc_pp.seperate_markers(markers_positions_no_large_movs, head_parts)
head_positions = dlc_pp.average_multiple_markers_to_single_one(updated_head_markers, flip=True)


# Plot the body positions ne

sv.image_sequence(globals(), 'frame', 'labeled_video_file')

global body_traj_x
body_traj_x = 0
global body_traj_y
body_traj_y = 0
global head_traj_x
head_traj_x = 0
global head_traj_y
head_traj_y = 0


def update_body_trajectory(f):
    global body_traj_x
    global body_traj_y
    body_traj_x = body_positions[:f, 0]
    body_traj_y = body_positions[:f, 1]
    return body_positions[:f, :]


def update_head_trajectory(f):
    global head_traj_x
    global head_traj_y
    head_traj_x = head_positions[:f, 0]
    head_traj_y = head_positions[:f, 1]
    return head_positions[:f, :]

body_traj = None
tr.connect_repl_var(globals(), 'frame', 'body_traj', 'update_body_trajectory')

head_traj = None
tr.connect_repl_var(globals(), 'frame', 'head_traj', 'update_head_trajectory')

osv.graph(globals(), 'body_traj_y', 'body_traj_x')
osv.graph(globals(), 'head_traj_y', 'head_traj_x')

