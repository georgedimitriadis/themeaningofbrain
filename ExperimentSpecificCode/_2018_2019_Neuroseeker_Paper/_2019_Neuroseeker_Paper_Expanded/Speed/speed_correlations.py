

from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import sem, t
from sklearn.decomposition import pca
import pickle

from npeet.lnc import MI

from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from idtxl.visualise_graph import plot_network

import one_shot_viewer as osv
import sequence_viewer as sv
import transform as tr
import slider as sl

from BrainDataAnalysis import binning
from BehaviorAnalysis import dlc_post_processing as dlc_pp
from BrainDataAnalysis import neuroseeker_specific_functions as ns_funcs

from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const

#  -------------------------------------------------
#  GET FOLDERS
#  -------------------------------------------------
dlc_folder = r'D:\Data\George\AK_33.1\2018_04_30-11_38\Analysis\Deeplabcut'
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-05-07')

date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')
spikes_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis', 'Denoised',
                     'Kilosort')
mutual_information_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis',
                                 'Results', 'MutualInformation')

num_of_frames_to_average = 0.25/(1/120)