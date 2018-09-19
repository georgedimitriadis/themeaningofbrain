
import numpy
import matplotlib.pyplot as pyplot
from BrainDataAnalysis import neuroseeker_specific_functions as ns_funcs

import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.events_sync_funcs as sync_funcs


CAMERA_TTL_PULSES_TIMEPOINT_PERIOD = 158

data_folders = {1: r'Z:\g\George\DataAndResults\Experiments\Awake\NeuroSeeker\AK_31.4\2018_04_02-13_15',
                2: r'Z:\g\George\DataAndResults\Experiments\Awake\NeuroSeeker\AK_31.4\2018_04_02-13_47',
                3: r'Z:\g\George\DataAndResults\Experiments\Awake\NeuroSeeker\AK_31.4\2018_04_03-11_07',
                4: r'Z:\g\George\DataAndResults\Experiments\Awake\NeuroSeeker\AK_31.4\2018_04_03-11_37',
                5: r'Z:\g\George\DataAndResults\Experiments\Awake\NeuroSeeker\AK_31.4\2018_04_04-10_05',
                6: r'Z:\g\George\DataAndResults\Experiments\Awake\NeuroSeeker\AK_31.4\2018_04_05-13_31',
                7: r'Z:\g\George\DataAndResults\Experiments\Awake\NeuroSeeker\AK_31.4\2018_04_05-14_11',
                8: r'Z:\g\George\DataAndResults\Experiments\Awake\NeuroSeeker\AK_31.4\2018_04_07-09_36',
                10: r'Z:\g\George\DataAndResults\Experiments\Awake\NeuroSeeker\AK_31.4\2018_04_09-11_36'}


camera_pulses, beam_breaks = sync_funcs.generate_events_from_sync_file(data_folders[10], clean=True,
                                                                       cam_ttl_pulse_period=
                                                                       CAMERA_TTL_PULSES_TIMEPOINT_PERIOD)
