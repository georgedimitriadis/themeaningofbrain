
import numpy
import matplotlib.pyplot as pyplot
from BrainDataAnalysis import neuroseeker_specific_functions as ns_funcs
from os.path import join
import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.events_sync_funcs as sync_funcs


CAMERA_TTL_PULSES_TIMEPOINT_PERIOD = 158


base_save_folder = r'Z:\g\George\DataAndResults\Experiments\Awake\NeuroSeeker'
rat_folder = r'AK_33.1'

date_folders = {1: r'2018_04_23-11_49',
                2: r'2018_04_24-10_12',
                3: r'2018_04_25-11_10',
                4: r'2018_04_26-12_13',
                5: r'2018_04_27-09_44',
                6: r'2018_04_28-19_20',
                7: r'2018_04_29-18_18',
                8: r'2018_04_30-11_38',
                9: r'2018_05_01-11_08',
                10: r'2018_05_02-09_27',
                11: r'2018_05_05-16_44',
                12: r'2018_05_06-14_16',
                13: r'2018_05_07-14_01',
                14: r'2018_05_08-13_12',
                15: r'2018_05_09-12_26',
                16: r'2018_05_10-15_57'}


data_folder = join(base_save_folder, rat_folder, date_folders[16])

camera_pulses, beam_breaks = sync_funcs.generate_events_from_sync_file(data_folder, clean=True,
                                                                       cam_ttl_pulse_period=
                                                                       CAMERA_TTL_PULSES_TIMEPOINT_PERIOD)

