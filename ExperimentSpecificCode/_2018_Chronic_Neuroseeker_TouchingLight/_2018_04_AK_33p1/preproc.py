
import numpy
import matplotlib.pyplot as pyplot
from BrainDataAnalysis import neuroseeker_specific_functions as ns_funcs
from os.path import join
import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.events_sync_funcs as sync_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const

date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')

camera_pulses, beam_breaks = sync_funcs.generate_events_from_sync_file(data_folder, clean=True,
                                                                       cam_ttl_pulse_period=
                                                                       const.CAMERA_TTL_PULSES_TIMEPOINT_PERIOD)

