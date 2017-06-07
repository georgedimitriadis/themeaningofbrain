
from GUIs.Kilosort import cleaning_kilosorting_results as clean
from os.path import join

base_folder = r'F:\Data\George\Projects\SpikeSorting\Neuroseeker\Neuroseeker_Chronic_Rat_22.1\2017_05_26\13_28_10\Analysis\Kilosort'
data_folder = r'F:\Data\George\Projects\SpikeSorting\Neuroseeker\Neuroseeker_Chronic_Rat_22.1\2017_05_26\13_28_10\Data'
binary_data_filename = join(data_folder, r'2017_05_26T13_28_10_Amp_S16_LP3p5KHz_uV.bin')

number_of_channels_in_binary_file = 1440


clean.cleanup_kilosorted_data(base_folder, number_of_channels_in_binary_file, binary_data_filename, generate_all=False,
                            overwrite_avg_spike_template_file=False, overwrite_template_marking_file=False, freq=20000,
                            time_points=100, figure_id=0, timeToPlot=None)
