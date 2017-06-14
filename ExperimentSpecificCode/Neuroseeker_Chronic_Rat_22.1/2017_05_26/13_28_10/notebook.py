
from GUIs.Kilosort import clean_kilosort_templates as clean
from os.path import join

base_folder = r'F:\Data\George\Projects\SpikeSorting\Neuroseeker\Neuroseeker_Chronic_Rat_22.1\2017_05_26\13_28_10\Analysis\Kilosort'
data_folder = r'F:\Data\George\Projects\SpikeSorting\Neuroseeker\Neuroseeker_Chronic_Rat_22.1\2017_05_26\13_28_10\Data'
binary_data_filename = join(data_folder, r'2017_05_26T13_28_10_Amp_S16_LP3p5KHz_uV.bin')

probe_info_folder = r'E:\George\Python35Projects\TheMeaningOfBrain\Layouts\Probes\Neuroseeker'
probe_connected_channels_file = r'neuroseeker_connected_channels_chronic_rat_22p1.npy'

time_points = 100
sampling_frequency = 20000

number_of_channels_in_binary_file = 1440


clean.cleanup_kilosorted_data(base_folder, number_of_channels_in_binary_file, binary_data_filename,
                              probe_info_folder, probe_connected_channels_file, sampling_frequency=20000)





