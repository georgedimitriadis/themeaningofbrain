import sys
from IO import lynxio as lio
import pylab as pl
import numpy as np
import pandas as pd
import os
from BrainDataAnalysis import ploting_functions as pf
from BrainDataAnalysis import timelocked_analysis_functions as tf
from BrainDataAnalysis import filters as filters
import scipy.signal as signal
import scipy as sp
from Layouts.Grids import grids as grids

f_sampling = 32556
f_hp_cutoff = 500
f_lp_cutoff = 1000
f_subsample = 2000
f_mua_lp_cuttof = 2000
f_mua_subsample = 4000
ADBitVolts = 0.0000000305

folder = r'D:\Data\George\Projects\SpikeSorting\AnalysisResults\Donders_B14R9\2015-07-10_12-43-23_B14R9_64ECoG_32Atlas_AwakeNoStim_HS1Ref1HS3Ref4'
memap_folder = r'D:\Data\George\Projects\SpikeSorting\AnalysisResults\Donders_B14R9\Analysis\Other'

ecog_bad_channels = [14, 30, 31, 34, 35, 36, 37, 38, 39, 40, 41, 45, 46, 48, 49, 53, 55, 57, 60, 61, 62, 63]
probe_bad_channels = [13, 14, 15, 16, 22, 25, 29, 30]



#----------Data generation-----------------
data = lio.read_all_csc(folder,  assume_same_fs=False, memmap=True, memmap_folder=memap_folder, save_for_spikedetekt=False, channels_to_save=None, return_sliced_data=False)
pl.save(os.path.join(memap_folder, 'B14R9_raw.npy'), data)

data_ecog = data[:64,:]
data_probe = data[64:,:]


data_probe_hp = pl.memmap(os.path.join(memap_folder,'data_probe_hp.dat'), dtype='int16', mode='w+', shape=pl.shape(data_probe))
for i in pl.arange(0, pl.shape(data_probe)[0]):
    data_probe_hp[i,:] = filters.high_pass_filter(data_probe[i,:], Fsampling=f_sampling, Fcutoff=f_hp_cutoff)
    data_probe_hp.flush()
    print(i)
pl.save(os.path.join(memap_folder, 'data_probe_hp.npy'), data_probe_hp)


shape_data_ss = (pl.shape(data_ecog)[0], pl.shape(data_ecog)[1]/int(f_sampling/f_subsample))
data_ecog_lp_ss = pl.memmap(os.path.join(memap_folder, 'data_ecog_lp_ss.dat'), dtype='int16', mode='w+', shape=shape_data_ss)
for i in pl.arange(0, pl.shape(data_ecog)[0]):
    data_ecog_lp_ss[i,:] = signal.decimate(filters.low_pass_filter(data_ecog[i,:], Fsampling=f_sampling, Fcutoff=f_lp_cutoff), int(f_sampling/f_subsample))
    data_ecog_lp_ss.flush()
    print(i)
pl.save(os.path.join(memap_folder, 'data_ecog_lp_ss.npy'), data_ecog_lp_ss)