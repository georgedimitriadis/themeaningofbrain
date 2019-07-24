
"""
This needs to run in the dlc_windowsGPU environment (an environment that has deeplabcut and tensorflow supporting GPU)

Also deeplabcut requires the Tk backend so remove the PyQt5 backend commands from the console starting commands
"""

from os.path import join
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p2 import constants as const
import deeplabcut


# ----------------------------------------------------------------------------------------------------------------------
# FOLDERS NAMES
date = 8
base_dlc_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                       'Analysis', 'Deeplabcut')
base_projects_folder = join(base_dlc_folder, 'projects')

cropped_video_filename = join(base_dlc_folder, 'BonsaiCroping', 'Croped_video.avi')

# ----------------------------------------------------------------------------------------------------------------------
config_path = deeplabcut.create_new_project(project='V1', experimenter='', videos=[cropped_video_filename],
                                            working_directory=base_projects_folder, copy_videos=True)

# Use the line below to 'reload the existing project
config_path = join(base_projects_folder, 'V1--2019-06-30', 'config.yaml')

# Edit the config.yaml file
deeplabcut.extract_frames(config_path, 'manual')

deeplabcut.label_frames(config_path)

deeplabcut.check_labels(config_path)

deeplabcut.create_training_dataset(config_path)

deeplabcut.train_network(config_path, gputouse=1)

deeplabcut.evaluate_network(config_path, plotting=True)

deeplabcut.analyze_videos(config_path, [cropped_video_filename], gputouse=1,
                          shuffle=1, save_as_csv=False, videotype='avi')

deeplabcut.create_labeled_video(config_path, [cropped_video_filename])
