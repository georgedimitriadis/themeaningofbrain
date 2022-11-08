
"""
The pipeline for spikesorting this dataset
"""

import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt


from spikesorting_tsne import preprocessing_kilosort_results as preproc_kilo, io_with_cpp as tsne_io, tsne as tsne, \
    visualization as viz
from BrainDataAnalysis.Spike_Sorting import positions_on_probe as spp
from spikesorting_tsne_guis import clean_kilosort_templates as clean


from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_25_AK_47p2 import constants as const

import sys
from io import StringIO

import transform as tr


# ----------------------------------------------------------------------------------------------------------------------
# FOLDERS NAMES
date = 8
binary_data_filename = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                            'Data', 'Amplifier_APs.bin')
analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                       'Analysis')
data_filename = join(analysis_folder, 'Data', 'Amplifier_APs_Denoised.bin')
kilosort_folder = join(analysis_folder, 'Kilosort')

tsne_exe_dir = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\win\x64\Release'
tsne_cortex_folder = join(analysis_folder, 'Tsne', 'Cortex')

sampling_freq = const.SAMPLING_FREQUENCY
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
#  STEP 1: RUN KILOSORT ON THE DATA
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# STEP 2: CLEAN DENOISED SPIKESORT (RIGHT AFTER KILOSORT)

# a) Create average of templates:
# To create averages of templates use cmd (because the create_data_cubes doesn't work when called from a REPL):
# Go to where the create_data_cubes.py is (in spikesort_tsne_guis/spikesort_tsen_guis) and run the following python command
# (you can use either the raw or the denoised data to create the average)
# python E:\Software\Develop\Source\Repos\spikesorting_tsne_guis\spikesorting_tsne_guis\create_data_cubes.py
#                                                                                original
#                                                                                "D:\\AK_47.2\2019_06_18-10_15\Analysis\Kilosort"
#                                                                                "D:\\AK_47.2\2019_06_18-10_15\Data\Amplifier_APs.bin"
#                                                                                1368
#                                                                                50
# (Use single space between parameters, not Enter like here)
# (Change the folders as appropriate for where the data is)

# b) Clean:
clean.cleanup_kilosorted_data(kilosort_folder,
                              number_of_channels_in_binary_file=const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE,
                              binary_data_filename=binary_data_filename,
                              prb_file=const.prb_file,
                              type_of_binary=const.BINARY_FILE_ENCODING,
                              order_of_binary='F',
                              sampling_frequency=20000,
                              num_of_shanks_for_vis=5)

# c) Remove some types
template_marking = np.load(join(kilosort_folder, 'template_marking.npy'))
print(len(np.argwhere(template_marking == 0)))
print(len(np.argwhere(template_marking == 1)))
print(len(np.argwhere(template_marking == 2)))
print(len(np.argwhere(template_marking == 3)))
print(len(np.argwhere(template_marking == 4)))
print(len(np.argwhere(template_marking == 5)))
print(len(np.argwhere(template_marking == 6)))
print(len(np.argwhere(template_marking == 7)))
template_marking[np.argwhere(template_marking == 5)] = 0
template_marking[np.argwhere(template_marking == 6)] = 0
np.save(join(kilosort_folder, 'template_marking.npy'), template_marking)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# STEP 3: CREATE TEMPLATE INFO OF ALL THE CLEANED TEMPLATES

# a) Create the positions of the templates on the probe (and have a look)
_ = spp.generate_probe_positions_of_templates(kilosort_folder)
bad_channel_positions = spp.get_y_spread_regions_of_bad_channel_groups(kilosort_folder, const.bad_channels)
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT)

# b) Create the template_info.df dataframe (or load it if you already have it)
# template_info = preproc_kilo.generate_template_info_after_cleaning(kilosort_folder, sampling_freq)
template_info = np.load(join(kilosort_folder, 'template_info.df'), allow_pickle=True)

# c) Make the spike info from the initial, cleaned, kilosort results
#spike_info = preproc_kilo.generate_spike_info_after_cleaning(kilosort_folder)
spike_info = np.load(join(kilosort_folder, 'spike_info_after_cleaning.df'), allow_pickle=True)
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info=template_info)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# LOOK AT THE FIRING RATES DISTRIBUTION

brain_regions = const.BRAIN_REGIONS
cortex = np.array([brain_regions['Cortex MPA'], brain_regions['CA1']]) / const.POSITION_MULT
hippocampus = np.array([brain_regions['CA1'], brain_regions['Thalamus LDVL']]) / const.POSITION_MULT
thalamus = np.array([brain_regions['Thalamus LDVL'], brain_regions['Zona Incerta']]) / const.POSITION_MULT

cort_cells = template_info[np.logical_and(template_info['position Y'] < cortex[0], template_info['position Y'] > cortex[1])]
hipp_cells = template_info[np.logical_and(template_info['position Y'] < hippocampus[0], template_info['position Y'] > hippocampus[1])]
thal_cells = template_info[np.logical_and(template_info['position Y'] < thalamus[0], template_info['position Y'] > thalamus[1])]


plt.hist(cort_cells['firing rate'], bins=np.logspace(np.log10(0.001), np.log10(100), 50))
plt.gca().set_xscale("log")


# ----------------------------------------------------------------------------------------------------------------------
# STEP 4: SEE TEMPLATES AGAIN AND DECIDE WHAT TO THROW AWAY
template_info = np.load(join(kilosort_folder, 'template_info.df'), allow_pickle=True)
avg_templates = np.load(join(kilosort_folder, 'avg_spike_template.npy'), allow_pickle=True)

spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS, const.POSITION_MULT,
                                     template_info=template_info)

f = plt.figure(0)
old_stdout = sys.stdout
global previous_template_number
previous_template_number = -1
global result
result = StringIO()
template_number = 0


def show_average_template(figure):
    global previous_template_number
    global result
    sys.stdout = result
    string = result.getvalue()
    new = string[-200:]

    try:
        template_number = int(new[new.find('Template number'): new.find('Template number')+22][18:22])
        if template_number != previous_template_number:
            template = template_info[template_info['template number'] == template_number]
            figure.clear()
            ax = figure.add_subplot(111)
            try:
                ax.plot(np.squeeze(avg_templates[template.index.values]).T)
            except:
                pass
        previous_template_number = template_number
        figure.suptitle('Template = {}, with {} number of spikes'.format(str(template_number),
                                                                         str(template['number of spikes'].values[0])))
    except:
        template_number = None
    return template_number


tr.connect_repl_var(globals(), 'f', 'template_number', 'show_average_template')
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
