import math
import os
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
from scipy import io
import IO.ephys as ephys
# import Utilities as ut
plt.rcParams['animation.ffmpeg_path'] ="C:\\Users\\KAMPFF-LAB_ANALYSIS4\\Downloads\\ffmpeg-20160116-git-d7c75a5-win64-static\\ffmpeg-20160116-git-d7c75a5-win64-static\\bin\\ffmpeg.exe"

#structure of folders and data files------------------------------------------------------------------------------------
base_folder = r'Z:\j\Joana Neto\Neuronexus32ch_clusters'

date = {1: '2014-11-13', 2: '2014-11-25', 3: '2017-02-02', 4: '2014-10-17'}

all_recordings_capture_times = {1: {'1': '19_01_55', '2': '18_48_11', '3': '14_59_40', '4': '15_35_31', '5': '18_05_50'},
                                2: {'1': '21_27_13', '2': '22_44_57', '3': '23_00_08', '4': '20_32_48'},
                                3: {'1': '14_38_11', '2': '15_03_44', '3': '15_49_35', '4': '16_57_16', '5': '17_18_46'},
                                4: {'1': '16_46_02', '2': '18_19_09'}}

recordings_cluster = {1: [ '3', '4', '5'],
                      2: ['1', '2', '3', '4'],
                      3: ['1', '2', '3', '4', '5'],
                      4: ['1', '2']}

#cluster_indices_good = {1: {'1': [46, 186, 252, 202], '2': [61, 2, 27, 23, 42, 54], '3': [23, 19, 37, 83, 91, 49, 40, 78, 85, 49, 52, 9, 5]},
                        #2:{'1': [], '2': []}}


#channel numbers for pedot and pristine electrodes----------------------------------------------------------------------
pedot_all= [22,29,3,23,18,4,17,19,5,16,20,6,30,21,7,31]
pristine_all= np.array([2,9,28,13,8,27,12,14,26,11,15,25,10,1,24,0])


#choose one recording for analysis--------------------------------------------------------------------------------------
surgery = 3
pick_recording = '1' #we need to select the recording

recordings = recordings_cluster[surgery]
#clusterSelected = cluster_indices_good[surgery][pick_recording]
date = date[surgery]
data_folder = os.path.join(base_folder + '\\' + date, 'Data')
analysis_folder = os.path.join(base_folder + '\\' + date, 'Analysis')
recordings_capture_times = all_recordings_capture_times[surgery]


#variables to open data-------------------------------------------------------------------------------------------------
num_of_points_in_spike_trig_ivm = 128
num_of_points_for_padding = 2 * num_of_points_in_spike_trig_ivm

inter_spike_time_distance = 30

num_ivm_channels = 32
amp_dtype = np.uint16

sampling_freq = 20000#neuronexus6
#sampling_freq = 30000#neuronexus5

high_pass_freq = 250
filtered_data_type = np.float64

types_of_data_to_load = {'t': 'ivm_data_filtered_cell{}.dat',
                         'f': 'ivm_data_filtered_cell{}',
                         'c': 'ivm_data_filtered_continous_cell{}.dat',
                         'k': 'ivm_data_filtered_klusta_spikes_cell{}.dat',
                         'm': 'ivm_data_raw_cell{}.dat',
                         }
types_of_sorting = {'s': 'goodcluster_{}_'}



#filter for extracellular recording-------------------------------------------------------------------------------------
def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=20000.0,passFreq=100.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b, a, data)



#import all kilosort clusters [cluster index, number of spikes, spike times]--------------------------------------------
spike_template_folder = os.path.join(base_folder + '\\' + date + '\\' + 'Datakilosort' + '\\' + 'amplifier' + date + 'T'
                                     + all_recordings_capture_times[surgery][pick_recording])
spike_templates = np.load(os.path.join(spike_template_folder + '\\' + 'spike_clusters.npy'))
spike_templates = np.reshape(spike_templates, ((len(spike_templates)),1))
spike_times = np.load(os.path.join(spike_template_folder + '\\' + 'spike_times.npy'))


kilosort_units = {}
for i in np.arange(len(spike_templates)):
    cluster = spike_templates[i][0]
    if cluster in kilosort_units:
        kilosort_units[cluster] = np.append(kilosort_units[cluster], i)
    else:
        kilosort_units[cluster] = i

cluster_info = pd.DataFrame(columns=['Cluster', 'Num_of_Spikes', 'Spike_Indices'])
cluster_info = cluster_info.set_index('Cluster')
cluster_info['Spike_Indices'] = cluster_info['Spike_Indices'].astype(list)

for g in kilosort_units.keys():
    if np.size(kilosort_units[g]) == 1:
        kilosort_units[g] = [kilosort_units[g]]
    cluster_name = str(g)
    cluster_info.set_value(cluster_name, 'Num_of_Spikes', len(kilosort_units[g]))
    cluster_info.set_value(cluster_name, 'Spike_Indices', kilosort_units[g])


#get index of good clusters---------------------------------------------------------------------------------------------
def find_good_clusters(spike_template_folder):
    # look at which clusters are marked as 'good' in phy GUI
    clust_file_name = os.path.join(spike_template_folder,"cluster_groups.csv")
    clust_ids = np.genfromtxt(clust_file_name,dtype= 'str')
    clust_ids = clust_ids[1:,] # skip header

    good_clusters = list()
    mua_clusters = list()
    noise_clusters = list()

    for i in np.arange(len(clust_ids)):
        if clust_ids[i,1]=='good':
            #print(clust_ids[i,0])
            good_clusters.append(int(clust_ids[i,0]))
        elif clust_ids[i,1]=='mua':
            mua_clusters.append(int(clust_ids[i,0]))
        elif clust_ids[i,1]=='noise':
            noise_clusters.append(int(clust_ids[i,0]))

    n_spikes = list()
    for i in good_clusters:
        n_spikes.append(len(kilosort_units[i]))

    good_cluster_spikecount = np.column_stack((good_clusters, n_spikes))
    np.savetxt(os.path.join(spike_template_folder, "spike_count_for_each_good_cluster.csv"), good_cluster_spikecount,
               delimiter=",", fmt='%10.5f')

    n_good_clusters = len(good_clusters)
    n_mua_clusters = len(mua_clusters)
    n_noise_clusters = len(noise_clusters)

    return good_clusters,n_good_clusters,n_mua_clusters,mua_clusters, n_noise_clusters, noise_clusters


[good_clusters,n_good_clusters,n_mua_clusters,mua_clusters, n_noise_clusters, noise_clusters] = find_good_clusters(spike_template_folder)
total = n_good_clusters + n_mua_clusters + n_noise_clusters

clusterSelected = good_clusters


# Generate the (channels x time_points x spikes) high passed extracellular recordings datasets for good clusters--------
all_cells_ivm_filtered_data = {}
data_to_load = 't'
cluster_to_load = 's'
passFreq = high_pass_freq

raw_data_file_ivm = os.path.join(data_folder + '\\' + 'amplifier' + date + 'T' + all_recordings_capture_times[surgery][pick_recording] + '.bin')
raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)

for clus in np.arange(0, len(clusterSelected)):
    num_of_spikes = len(kilosort_units[clusterSelected[clus]])

    shape_of_filt_spike_trig_ivm = ((num_ivm_channels,
                                 num_of_points_in_spike_trig_ivm,
                                 num_of_spikes))

    ivm_data_filtered = np.memmap(os.path.join(analysis_folder, types_of_sorting[cluster_to_load].format(clusterSelected[clus]) + types_of_data_to_load[data_to_load].format(pick_recording)),
                                               dtype=filtered_data_type,
                                               mode='w+',
                                               shape=shape_of_filt_spike_trig_ivm)

    for spike in np.arange(0, num_of_spikes):
        trigger_point = spike_times[kilosort_units[clusterSelected[clus]][spike]]
        start_point = int(trigger_point - (num_of_points_in_spike_trig_ivm + num_of_points_for_padding)/2)
        if start_point < 0:
            break
        end_point = int(trigger_point + (num_of_points_in_spike_trig_ivm + num_of_points_for_padding)/2)
        if end_point > raw_data_ivm.shape()[1]:
            break
        temp_unfiltered = raw_data_ivm.dataMatrix[:, start_point:end_point]
        temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
        temp_filtered = highpass(temp_unfiltered, F_HIGH = (sampling_freq/2)*0.95, sampleFreq = sampling_freq, passFreq = high_pass_freq)
        temp_filtered = temp_filtered[:, int(num_of_points_for_padding / 2):-int(num_of_points_for_padding / 2)]
        ivm_data_filtered[:, :, spike] = temp_filtered
del ivm_data_filtered


# Load the extracellular recording cut data from the .dat files on hard disk onto memmaped arrays-----------------------
all_cells_ivm_filtered_data = {}
data_to_load = 't'
cluster_to_load = 's'

for clus in np.arange(0, len(clusterSelected)):
    num_of_spikes = len(kilosort_units[clusterSelected[clus]])
    if data_to_load == 't':
        shape_of_filt_spike_trig_ivm = (num_ivm_channels,
                                    num_of_points_in_spike_trig_ivm,
                                    num_of_spikes)
        time_axis = np.arange(-num_of_points_in_spike_trig_ivm/(2*sampling_freq),
                  num_of_points_in_spike_trig_ivm/(2*sampling_freq),
                  1/sampling_freq)

    ivm_data_filtered = np.memmap(os.path.join(analysis_folder, types_of_sorting[cluster_to_load].format(clusterSelected[clus]) + types_of_data_to_load[data_to_load].format(pick_recording)),
                                dtype=filtered_data_type,
                                mode='r',
                                shape=shape_of_filt_spike_trig_ivm)

    all_cells_ivm_filtered_data[clusterSelected[clus]] = ivm_data_filtered


#PLOT
#----------------------------------------------------------------------------------------------------------------------

# create folder to save figures ---------------------------------------------------------------------------------------
fig_folder = os.path.join(analysis_folder+ '\\'+ 'figures' +  '\\' + 'amplifier' + date + 'T' +
  all_recordings_capture_times[surgery][pick_recording])

if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

#Autocorrelagram for each cluster--------------------------------------------------------------------------------------
def crosscorrelate_spike_trains(spike_times_train_1, spike_times_train_2, lag=None):
    if spike_times_train_1.size < spike_times_train_2.size:
        if lag is None:
            lag = np.ceil(10 * np.mean(np.diff(spike_times_train_1)))
        reverse = False
    else:
        if lag is None:
            lag = np.ceil(20 * np.mean(np.diff(spike_times_train_2)))
        spike_times_train_1, spike_times_train_2 = spike_times_train_2, spike_times_train_1
        reverse = True

    differences = np.array([])
    for k in np.arange(0, spike_times_train_1.size):
        differences = np.append(differences, spike_times_train_1[k] - spike_times_train_2[np.nonzero(
            (spike_times_train_2 > spike_times_train_1[k] - lag)
             & (spike_times_train_2 < spike_times_train_1[k] + lag)
             & (spike_times_train_2 != spike_times_train_1[k]))])
    if reverse is True:
        differences = -differences
    norm = np.sqrt(spike_times_train_1.size * spike_times_train_2.size)
    return differences, norm

#save plots, spike times during one recording and autocorrelogram-------------------------------------------------------
lag = 50
for pick_cluster in clusterSelected:
    times_cluster = np.reshape(spike_times[kilosort_units[pick_cluster]], (spike_times[kilosort_units[pick_cluster]].shape[0]))
    plt.figure()
    plt.vlines(times_cluster, ymin = 0,ymax = 1, label = np.str(pick_cluster))
    plt.ylim(-4, 4)
    plt.title('Cluster ' + str(pick_cluster))
    fig_filename = os.path.join(fig_folder, types_of_sorting[cluster_to_load].format(pick_cluster)
                                     + 'spiketimes.png')
    plt.savefig(fig_filename)
    diffs, norm = crosscorrelate_spike_trains((times_cluster/(sampling_freq/1000)).astype(np.int64),(times_cluster/(sampling_freq/1000)).astype(np.int64),lag=50)
    #hist, edges = np.histogram(diffs)
    plt.figure()
    plt.hist(diffs, bins=101, normed=False, range=(-50,50), align ='mid', label = np.str(pick_cluster)) #if normed = True, the probability density function at the bin, normalized such that the integral over the range is 1.
    plt.xlim(-lag, lag)
    #plt.legend()
    plt.title('Cluster ' + str(pick_cluster))
    fig_filename = os.path.join(fig_folder, types_of_sorting[cluster_to_load].format(pick_cluster)+ 'autocorrelogram.png')
    plt.savefig(fig_filename)
    #plt.close()



# Colormap from Pylyb--------------------------------------------------------------------------------------------------
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap



# Plot a line at time x------------------------------------------------------------------------------------------------
def triggerline(x, **kwargs):
    if x is not None:
        ylim = plt.ylim()
        plt.vlines(x,ylim[0],ylim[1],alpha=0.4)



# Plot 32channel averages overlaid color coded--------------------------------------------------------------------------
voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000

def plot_average_extra(all_cells_ivm_filtered_data, yoffset=0):
    origin_cmap= plt.get_cmap('hsv')
    shrunk_cmap = shiftedColorMap(origin_cmap, start=0.6, midpoint=0.7, stop=1, name='shrunk')
    cm=shrunk_cmap
    cNorm=colors.Normalize(vmin=0,vmax=32)
    scalarMap= cmx.ScalarMappable(norm=cNorm,cmap=cm)
    sites_order_geometry= [0,31,24,7,1,21,10,30,25,6,15,20,11,16,26,5,14,19,12,17,27,4,8,18,13,23,28,3,9,29,2,22]
    for clus in np.arange(0, len(clusterSelected)):
        num_of_spikes = len(kilosort_units[clusterSelected[clus]])
        extra_average_V = np.average(all_cells_ivm_filtered_data[clusterSelected[clus]][:,:,:],axis=2)
        num_samples=extra_average_V.shape[1]
        sample_axis= np.arange(-(num_samples/2),(num_samples/2))
        time_axis= sample_axis/sampling_freq
        extra_average_microVolts = extra_average_V * scale_uV * voltage_step_size
        plt.figure()
        for m in np.arange(np.shape(extra_average_microVolts)[0]):
            colorVal=scalarMap.to_rgba(np.shape(extra_average_microVolts)[0]-m)
            plt.plot(time_axis*scale_ms, extra_average_microVolts[sites_order_geometry[m],:].T, color=colorVal)
            plt.title('Cluster ' + str(clusterSelected[clus]))
            fig_filename = os.path.join(fig_folder, types_of_sorting[cluster_to_load].format(clusterSelected[clus])
                                     + 'AverageVoltagesOverlaid.pdf')
            plt.savefig(fig_filename)
            plt.xlim(-2, 2) #window 4ms
            plt.ylim(np.min(extra_average_microVolts)-yoffset, np.max(extra_average_microVolts)+yoffset)
            plt.ylabel('Voltage (\u00B5V)', fontsize=20)
            plt.xlabel('Time (ms)',fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            triggerline(0)
        plt.close()



plot_average_extra(all_cells_ivm_filtered_data, yoffset=0)


# Plot 32channel averages in space color coded--------------------------------------------------------------------------

voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000

def plot_average_extra_geometry(all_cells_ivm_filtered_data, yoffset=0):
    sites_order_geometry = [0,31,24,7,1,21,10,30,25,6,15,20,11,16,26,5,14,19,12,17,27,4,8,18,13,23,28,3,9,29,2,22]
    subplot_order = [2,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65]
    origin_cmap = plt.get_cmap('hsv')
    shrunk_cmap = shiftedColorMap(origin_cmap, start=0.6, midpoint=0.7, stop=1, name='shrunk')
    cm = shrunk_cmap
    cNorm = colors.Normalize(vmin = -4, vmax = 28)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    for clus in np.arange(0, len(clusterSelected)):
        num_of_spikes = len(kilosort_units[clusterSelected[clus]])
        plt.figure(clus+1)
        #plt.title('Cluster ' + str(clusterSelected[clus]))
        extra_average_microVolts= np.average(all_cells_ivm_filtered_data[clusterSelected[clus]][:,:,:],axis=2) * voltage_step_size * scale_uV
        num_samples=extra_average_microVolts.shape[1]
        sample_axis= np.arange(-(num_samples/2),(num_samples/2))
        time_axis= sample_axis/30000
        plt.title('Cluster ' + str(clusterSelected[clus]) + ','+ str(
            np.min(np.min(extra_average_microVolts, axis=1)) - yoffset) + ',' + str(
            np.max(np.max(extra_average_microVolts, axis=1)) + yoffset))
        for i in np.arange(32):
            plt.subplot(22,3,subplot_order [i])
            colorVal=scalarMap.to_rgba(31-i)
            plt.plot(time_axis*scale_ms, extra_average_microVolts[sites_order_geometry[i],:],color=colorVal)
            plt.ylim(np.min(np.min(extra_average_microVolts, axis=1))- yoffset, np.max(np.max(extra_average_microVolts, axis=1))+ yoffset)
            plt.xlim(-2, 2)
            plt.axis("OFF")
        fig_filename = os.path.join(fig_folder, types_of_sorting[cluster_to_load].format(clusterSelected[clus]) + 'AverageVoltagesSpace.pdf')
        plt.savefig(fig_filename)
        #plt.close()



plot_average_extra_geometry(all_cells_ivm_filtered_data, yoffset=0)



#-----------------------------------------------------------------------------------------------------------------------
# For each cluster, for the averages of the 32channels compute: P2P, MIN, MAX

voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000

def peaktopeak(all_cells_ivm_filtered_data, windowSize=60):

    for clus in np.arange(0, len(clusterSelected)):
        extra_average_V = np.average(all_cells_ivm_filtered_data[clusterSelected[clus]][:,:,:],axis=2) * voltage_step_size
        NumSamples = extra_average_V.shape[1]
        extra_average_microVolts = extra_average_V * scale_uV
        NumSites = np.size(extra_average_microVolts,axis = 0)
        lowerBound = int(NumSamples/2.0-windowSize/2.0)
        upperBound = int(NumSamples/2.0+windowSize/2.0)

        argminima = np.zeros(NumSites)
        for m in range(NumSites):
            argminima[m] = np.argmin(extra_average_microVolts[m][lowerBound:upperBound])+lowerBound
        #argminima = argminima/30 #convert to ms

        argmaxima = np.zeros(NumSites)
        for n in range(NumSites):
            argmaxima[n] = np.argmax(extra_average_microVolts[n][lowerBound:upperBound])+lowerBound
        #argmaxima = argmaxima/30 #convert to ms

        maxima = np.zeros(NumSites)
        for p in range(NumSites):
                maxima[p] = np.max(extra_average_microVolts[p][lowerBound:upperBound])

        minima = np.zeros(NumSites)
        for k in range(NumSites):
            minima[k] = np.min(extra_average_microVolts[k][lowerBound:upperBound])

        p2p = maxima-minima

        stdv_minimos = np.zeros(NumSites)
        stdv_maximos = np.zeros(NumSites)

        stdv = stats.sem(all_cells_ivm_filtered_data[clusterSelected[clus]][:,:,:],axis=2)
        stdv = stdv * voltage_step_size * scale_uV

        for b in range(NumSites):
           stdv_minimos[b] = stdv[b, argminima[b].astype(int)]
           stdv_maximos[b] = stdv[b, argmaxima[b].astype(int)]

        error = np.sqrt((stdv_minimos * stdv_minimos) + (stdv_maximos*stdv_maximos))


        np.save(os.path.join(analysis_folder,'stdv_minimos_EXTRA_Cluster'+ date + '_' + pick_recording +  '_' + str(clusterSelected[clus]) + '.npy'), stdv_minimos)
        np.save(os.path.join(analysis_folder,'stdv_maximos_EXTRA_Cluster'+ date + '_' + pick_recording +  '_' + str(clusterSelected[clus]) + '.npy'), stdv_maximos)
        np.save(os.path.join(analysis_folder,'stdv_average_EXTRA_Cluster'+ date + '_' + pick_recording +  '_' + str(clusterSelected[clus]) + '.npy'), stdv)
        np.save(os.path.join(analysis_folder,'error_EXTRA_Cluster'+ date + '_' + pick_recording +  '_' + str(clusterSelected[clus]) + '.npy'), error)
        np.save(os.path.join(analysis_folder,'p2p_EXTRA_Cluster'+ date + '_' + pick_recording +  '_' + str(clusterSelected[clus]) + '.npy'), p2p)
        np.save(os.path.join(analysis_folder,'minima_EXTRA_Cluster'+ date + '_' + pick_recording +  '_' + str(clusterSelected[clus]) + '.npy'), minima)
        np.save(os.path.join(analysis_folder,'maxima_EXTRA_Cluster' + date + '_' + pick_recording +  '_' + str(clusterSelected[clus]) + '.npy'), maxima)
        np.save(os.path.join(analysis_folder,'argmaxima_EXTRA_Cluster'+ date + '_' + pick_recording +  '_' +  str(clusterSelected[clus]) + '.npy'), argmaxima)
        np.save(os.path.join(analysis_folder,'argminima_EXTRA_Cluster'+ date + '_' + pick_recording +  '_' + str(clusterSelected[clus]) + '.npy'), argminima)

    return argmaxima, argminima, maxima, minima, p2p


peaktopeak(all_cells_ivm_filtered_data, windowSize=60)




#For each cluster print the minP2P, maxP2P, channel w the maxP2P, biggestP2Pchannels, pedot and pristine channels, ....

#os.path.join(spike_template_folder, "spike_count_for_each_good_cluster.csv")


f = open(os.path.join(spike_template_folder, "P2P_MIN_MAX_good_cluster.txt"),"w")

fig = plt.figure()
for pick_cluster in clusterSelected:
    filename_p2p = os.path.join(analysis_folder + '\\' + 'p2p_EXTRA_Cluster' + date + '_' + pick_recording +  '_' + str(pick_cluster) + '.npy')
    amplitudes = np.load(filename_p2p)
    min_ampl = amplitudes.min()
    max_ampl = amplitudes.max()
    channel = amplitudes.argmax()

    max_ampl_PEDOT = amplitudes[pedot_all].max()
    max_ampl_pristine = amplitudes[pristine_all].max()
    channel_PEDOT_pos = amplitudes[pedot_all].argmax()
    channel_pristine_pos = amplitudes[pristine_all].argmax()
    channel_PEDOT = pedot_all[channel_PEDOT_pos]
    channel_pristine = pristine_all[channel_pristine_pos]

    f.write('#------------------------------------------------------'+'\n')
    f.write('cluster_' + str(pick_cluster)+'\n')
    f.write('min_p2p ' + str(min_ampl)+'\n')
    f.write('max_p2p ' + str(max_ampl)+'\n')
    f.write('maxp2p_channel ' + str(channel)+'\n')
    select_indices_biggestAmp = np.where(amplitudes >= (amplitudes.max()/2))
    f.write('Channels_atleasthalfofmaxp2p ' + str(select_indices_biggestAmp)+'\n')
    Amp_select_indices = amplitudes[select_indices_biggestAmp]
    f.write('Amplitude_atleasthalfofmaxp2p ' + str(Amp_select_indices)+'\n')
    thebiggestchannel_pedot = np.intersect1d(channel, pedot_all)
    f.write('isthemaxp2p_channel_PEDOT?' + str(thebiggestchannel_pedot)+'\n')
    thebiggestchannel_pristine = np.intersect1d(channel,pristine_all)
    f.write('isthemaxp2p_channel_Pristine? ' + str(thebiggestchannel_pristine)+'\n')
    select_pedot = np.intersect1d (select_indices_biggestAmp, pedot_all)
    select_pristine = np.intersect1d (select_indices_biggestAmp, pristine_all)
    f.write('atleasthalfofmaxp2p_channels_PEDOT ' + str(select_pedot)+'\n')
    f.write('atleasthalfofmaxp2p_channels_Pristine ' + str(select_pristine)+'\n')
    if select_pristine.any():
        min_biggestchannel_pristine = amplitudes[select_pristine].min()
        channel_pristine_min_thebiggestchannels = select_pristine[amplitudes[select_pristine].argmin()]
    if select_pedot.any():
        min_biggestchannel_pedot = amplitudes[select_pedot].min()
        channel_pedot_min_thebiggestchannels = select_pedot[amplitudes[select_pedot].argmin()]
    f.write('min_atleasthalfofmaxp2p_channels_PEDOT ' + str(min_biggestchannel_pedot) + '\n')
    f.write('min_atleasthalfofmaxp2p_channels_Pristine ' + str(min_biggestchannel_pristine) + '\n')
    f.write('channel_min_atleasthalfofmaxp2p_channels_PEDOT ' + str(channel_pedot_min_thebiggestchannels) + '\n')
    f.write('channel_min_atleasthalfofmaxp2p_channels_Pristine ' + str(channel_pristine_min_thebiggestchannels) + '\n')
    f.write('bestPEDOTchannel ' + ','+ 'bestPristinechannel' + ','+ 'bestP2PPEDOT' + ','+ 'bestP2PPristine'+'\n')
    f.write(str(channel_PEDOT)+ ',' + str(channel_pristine)+ ','+ str(max_ampl_PEDOT)+ ','+ str(max_ampl_pristine)+'\n')
    f.write('bestPEDOTchannel ' + ','+ 'bestPristinechannel' + ','+ 'bestP2PPEDOT_min' + ','+ 'bestP2PPristine_min'+'\n')
    f.write(str(channel_pedot_min_thebiggestchannels)+ ',' + str(channel_pristine_min_thebiggestchannels)+ ','+ str(min_biggestchannel_pedot)+ ','+ str(min_biggestchannel_pristine)+'\n')
    f.write('#------------------------------------------------------'+'\n')

f.close()


#For each cluster print the maxP2P for the best ch pedot and pristine--------------------------------------------------

os.path.join(spike_template_folder, "spike_count_for_each_good_cluster.csv")
f = open(os.path.join(spike_template_folder, "P2P_MAX.txt"),"w")


f.write('bestPEDOTchannel ' + ','+ 'bestPristinechannel' + ','+ 'bestP2PPEDOT' + ','+ 'bestP2PPristine'+'\n')

fig = plt.figure()
for pick_cluster in clusterSelected:
    filename_p2p = os.path.join(analysis_folder + '\\' + 'p2p_EXTRA_Cluster' + date + '_' + pick_recording +  '_' + str(pick_cluster) + '.npy')
    amplitudes = np.load(filename_p2p)

    max_ampl_PEDOT = amplitudes[pedot_all].max()
    max_ampl_pristine = amplitudes[pristine_all].max()
    channel_PEDOT_pos = amplitudes[pedot_all].argmax()
    channel_pristine_pos = amplitudes[pristine_all].argmax()
    channel_PEDOT = pedot_all[channel_PEDOT_pos]
    channel_pristine = pristine_all[channel_pristine_pos]

    f.write(str(channel_PEDOT)+ ',' + str(channel_pristine)+ ','+ str(max_ampl_PEDOT)+ ','+ str(max_ampl_pristine)+'\n')
f.close()



#plot the waveforms of the best channel PEDOT versus Pristine-----------------------------------------------------------

for pick_cluster in clusterSelected:
    f, axarr = plt.subplots(2, sharex=True, sharey=True)
    filename_p2p = os.path.join(analysis_folder + '\\' + 'p2p_EXTRA_Cluster' + date + '_' + pick_recording +  '_' + str(pick_cluster) + '.npy')
    amplitudes = np.load(filename_p2p)
    select_indices_biggestAmp = np.array(np.where(amplitudes >= (amplitudes.max()/2)))

    if select_indices_biggestAmp.shape[1] <= 1:
        print('cluster ' + str(pick_cluster) + 'NOT GOOD')

    else:
        channel = amplitudes.argmax()
        select_pristine = np.intersect1d(select_indices_biggestAmp, pristine_all)
        select_pedot = np.intersect1d(select_indices_biggestAmp, pedot_all)
        best_channel_pedot = select_pedot[amplitudes[select_pedot].argmax()]
        best_channel_pridtine = select_pristine[amplitudes[select_pristine].argmax()]
        num_samples = all_cells_ivm_filtered_data[pick_cluster][:,:,:].shape[1]
        sample_axis = np.arange(-(num_samples/2),(num_samples/2))
        time_axis = sample_axis/sampling_freq
        num_of_spikes = len(kilosort_units[pick_cluster])
        subset = np.random.choice(np.arange(num_of_spikes), math.ceil(num_of_spikes * 1))
        extra_average_Pristine = np.average(all_cells_ivm_filtered_data[pick_cluster][best_channel_pridtine,:,:],axis=1) * voltage_step_size * scale_uV
        extra_average_PEDOT = np.average(all_cells_ivm_filtered_data[pick_cluster][best_channel_pedot,:,:],axis=1) * voltage_step_size * scale_uV

        #pristine best channel
        #axarr[0].plot(time_axis * scale_ms, all_cells_ivm_filtered_data[pick_cluster][best_channel_pridtine,:,subset].T * voltage_step_size * scale_uV, color='#b6b4b4')
        #axarr[0].set_title('cluster'+str(pick_cluster))
        #axarr[0].plot(time_axis * scale_ms, extra_average_Pristine, color='#5da4d6', linewidth=3)
     #pedot best channel
        #axarr[1].plot(time_axis * scale_ms, all_cells_ivm_filtered_data[pick_cluster][best_channel_pedot,:,subset].T * voltage_step_size * scale_uV, color='#b6b4b4')
        #axarr[1].plot(time_axis * scale_ms, extra_average_PEDOT, color='#ff4136', linewidth=3, label='%i'% num_of_spikes)

              #pristine best channel
        axarr[0].plot(time_axis * scale_ms, all_cells_ivm_filtered_data[pick_cluster][best_channel_pridtine,:,subset].T * voltage_step_size * scale_uV, color='#5da4d6', alpha=0.005)
        axarr[0].set_title('cluster'+str(pick_cluster))
        axarr[0].plot(time_axis * scale_ms, extra_average_Pristine, color='#5da4d6', linewidth=3)
        #pedot best channel
        axarr[1].plot(time_axis * scale_ms, all_cells_ivm_filtered_data[pick_cluster][best_channel_pedot,:,subset].T * voltage_step_size * scale_uV, color='#ff4136', alpha = 0.005)
        axarr[1].plot(time_axis * scale_ms, extra_average_PEDOT, color='#ff4136', linewidth=3, label='%i'% num_of_spikes)

        plt.xlim(-2, 2)
        f.subplots_adjust(hspace=0.2)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        fig_filename = os.path.join(fig_folder, types_of_sorting[cluster_to_load].format(pick_cluster) + 'BestChannelsVoltagePEDOTvsPristine.pdf')
        plt.savefig(fig_filename)
        plt.close()



#plot averages of the best channel PEDOT versus Pristine overlaid-------------------------------------------------------

number_clusters = len(clusterSelected)
ylim = 500
fig = plt.figure()

for pick_cluster in clusterSelected:
    filename_p2p = os.path.join(analysis_folder + '\\' + 'p2p_EXTRA_Cluster' + date + '_' + pick_recording + '_' + str(pick_cluster) + '.npy')
    amplitudes = np.load(filename_p2p)
    select_indices_biggestAmp = np.array(np.where(amplitudes >= (amplitudes.max()/2)))

    if select_indices_biggestAmp.shape[1] <= 1:
        print('cluster ' + str(pick_cluster) + 'NOT GOOD')

    else:
        channel = amplitudes.argmax()
        select_pristine = np.intersect1d(select_indices_biggestAmp, pristine_all)
        select_pedot = np.intersect1d(select_indices_biggestAmp, pedot_all)
        best_channel_pedot = select_pedot[amplitudes[select_pedot].argmax()]
        best_channel_pridtine = select_pristine[amplitudes[select_pristine].argmax()]
        num_samples = all_cells_ivm_filtered_data[pick_cluster][:,:,:].shape[1]
        sample_axis = np.arange(-(num_samples/2),(num_samples/2))
        time_axis = sample_axis/sampling_freq
        num_of_spikes = len(kilosort_units[pick_cluster])
        subset = np.random.choice(np.arange(num_of_spikes), math.ceil(num_of_spikes * 1))
        extra_average_Pristine = np.average(all_cells_ivm_filtered_data[pick_cluster][best_channel_pridtine,:,:], axis=1) * voltage_step_size * scale_uV
        extra_average_PEDOT = np.average(all_cells_ivm_filtered_data[pick_cluster][best_channel_pedot,:,:], axis=1) * voltage_step_size * scale_uV

        plt.figure()
        plt.plot(time_axis * scale_ms, extra_average_Pristine, color='#5da4d6', label ='Pristine', linewidth=3)
        plt.plot(time_axis*scale_ms,extra_average_PEDOT,color='#ff4136', label ='PEDOT', linewidth=3)
        plt.title('Cluster ' + str(pick_cluster))
        plt.ylim(-ylim, ylim)
        plt.xlim(-2, 2)
        fig_filename = os.path.join(fig_folder, types_of_sorting[cluster_to_load].format(pick_cluster) + 'BestChannelsAverageVoltagePEDOTvsPristine.pdf')
        plt.savefig(fig_filename)
        plt.legend()
        plt.close()



#plot waveform averages where(amplitudes >= (amplitudes.max()/2);the bigger P2P amplitude channels----------------------

pedot_all = [22,29,3,23,18,4,17,19,5,16,20,6,30,21,7,31]
pristine_all = np.array([2,9,28,13,8,27,12,14,26,11,15,25,10,1,24,0])
sites_order_geometry = [0,31,24,7,1,21,10,30,25,6,15,20,11,16,26,5,14,19,12,17,27,4,8,18,13,23,28,3,9,29,2,22]
subplot_order = [2,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65]
yoffset = 100

for pick_cluster in clusterSelected:
    filename_p2p = os.path.join(analysis_folder + '\\' + 'p2p_EXTRA_Cluster' + date + '_' + pick_recording +  '_' + str(pick_cluster) + '.npy')
    filename_min = os.path.join(analysis_folder + '\\' +'\minima_EXTRA_Cluster' + date + '_' + pick_recording +  '_' +str(pick_cluster) + '.npy')
    amplitudes = np.load(filename_p2p)
    select_indices_biggestAmp = np.where(amplitudes >= (amplitudes.max()/2))
    channel = amplitudes.argmax()
    select_pristine = np.intersect1d(select_indices_biggestAmp, pristine_all)
    select_pedot = np.intersect1d(select_indices_biggestAmp, pedot_all)

    extra_average_microVolts = np.average(all_cells_ivm_filtered_data[pick_cluster][:,:,:], axis=2) * voltage_step_size * scale_uV
    num_samples = extra_average_microVolts.shape[1]
    sample_axis = np.arange(-(num_samples/2),(num_samples/2))
    time_axis = sample_axis/sampling_freq

    plt.figure()

    for i in np.arange(num_ivm_channels):
        plt.subplot(22, 3, subplot_order[i])
        if sites_order_geometry[i] in select_pedot:
            colorVal = 'r'
            alphanumber = 1
        elif sites_order_geometry[i] in select_pristine:
            colorVal = 'b'
            alphanumber = 1
        else:
            colorVal = 'grey'
            alphanumber = 0.1

        plt.plot(time_axis * scale_ms, extra_average_microVolts[sites_order_geometry[i],:],color=colorVal, linewidth = 1, alpha = alphanumber)
        #plt.ylim(np.min(np.min(extra_average_microVolts, axis=1)) - yoffset, np.max(np.max(extra_average_microVolts, axis=1)) + yoffset)
        plt.ylim(-300, 300)
        plt.xlim(-2, 2)
        plt.axis("OFF")
    plt.title('cluster'+str(pick_cluster), verticalalignment='top', horizontalalignment='right')
    fig_filename = os.path.join(fig_folder, types_of_sorting[cluster_to_load].format(pick_cluster) + 'ChannelsAverageVoltagePEDOTvsPristine.pdf')
    plt.savefig(fig_filename)
    plt.axis("OFF")
    plt.close()





#Quantify the quality of cluster----------------------------------------------------------------------------------------
#first we run in matlab:
#addpath(genpath('F:\sortingQuality-master\sortingQuality-master'))
#addpath('F:\sortingQuality-master\sortingQuality-master')
#addpath('F:\npy-matlab-master')
#[cgs, uQ, cR, isiV] = sqKilosort.computeAllMeasures('Z:\j\Joana Neto\Neuronexus32ch\2014-11-13\Datakilosort\amplifier2014-11-13T14_59_40')
#[cgs, uQ, cR, isiV] = sqKilosort.computeAllMeasures('Z:\j\Joana Neto\Neuronexus32ch\2017-02-02\Datakilosort\amplifier2017-02-02T15_49_35')

#Steve code to open quantification files from clusters
#Load  files
def flatten_list(list_to_flatten):
    """
    turn a list of lists into a single flat list
    :param list_to_flatten:
    :return:
    """
    flatten = [item for sublist in list_to_flatten for item in sublist]
    return flatten


def load_quality_outputs(path):
    quality = io.loadmat(path)
    cluster_groups = flatten_list(quality['cgs'].T)
    unit_quality = quality['uQ']
    isi_violations = quality['isiV'].T
    contamination_rate = quality['cR']
    return cluster_groups, isi_violations, contamination_rate, unit_quality


quality_file = os.path.join(spike_template_folder + '\\' + 'quality.mat')
cluster_groups, isi_violation, contamination_rate, unit_quality = load_quality_outputs(quality_file)


#plot clusters quality: isi violations vs contamination rate colored according unit quality-----------------------------
fig_folder = os.path.join(spike_template_folder+ '\\'+ 'Figures')

if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)


def get_colormaps(palette, min_val, max_val):
    from matplotlib.colors import BoundaryNorm
    # define the colormap
    cmap = plt.get_cmap(palette)

    # extract all colors from the Reds map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize and forcing 0 to be part of the colorbar!
    bounds = np.arange(min_val, max_val, 1)
    idx = np.searchsorted(bounds, 0)
    bounds = np.insert(bounds, idx, 0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def plot_cluster_quality_w_label(cluster_id, spike_templates, isi_violation,
                         contamination_rate, unit_quality,
                         color, zorder, s):

    cluster_ids_raw = np.unique(spike_templates)
    cid = np.where(cluster_ids_raw == cluster_id)
    contamination_rates = contamination_rate[cid]
    if np.isnan(contamination_rates) == True:
        contamination_rates = -1

    isi_violations = isi_violation[cid]
    if np.isnan(isi_violations) == True:
        isi_violations = -1

    unit_qualities = unit_quality[cid]
    if np.isnan(unit_qualities) == True:
        unit_qualities = -1


    cmap, norm = get_colormaps(color, 0, 100)
    plt.scatter(contamination_rates, isi_violations, cmap=cmap, norm=norm, c=unit_qualities, zorder=zorder, s=s)
    plt.text(contamination_rates, isi_violations, '%s' % (str(cluster_id)), size=15, zorder = 0, color='k', rotation=90)
    plt.xlabel('contamination rate')
    plt.ylabel('isi violations')


plt.figure()
for i in mua_clusters:
    plot_cluster_quality_w_label(i, spike_templates, isi_violation, contamination_rate, unit_quality,'Blues', zorder=0, s=200 )
for clus in good_clusters:
    plot_cluster_quality_w_label(clus, spike_templates, isi_violation, contamination_rate, unit_quality, 'Reds', zorder=0, s=100)
fig_filename = os.path.join(fig_folder + '\\' + 'quality.png')
plt.savefig(fig_filename)


#for 'Z:\\j\\Joana Neto\\Neuronexus32ch\\2014-11-25\\Datakilosort\\amplifier2014-11-25T21_27_13' we good_clusters = np.delete(good_clusters, np.where(good_clusters==139),0) because cluster 139 has a contamainatio rate of Nan

# plot the first quadrant-----------------------------------------------------------------------------------------------
plt.figure()
for i in mua_clusters:
    plot_cluster_quality_w_label(i, spike_templates, isi_violation, contamination_rate, unit_quality,'Blues', zorder=0, s=200 )
    plt.xlim(-0.2, 0.5)
    plt.ylim(-0.2, 0.5)
for clus in good_clusters:
    plot_cluster_quality_w_label(clus, spike_templates, isi_violation, contamination_rate, unit_quality, 'Reds', zorder=0, s=100)
fig_filename = os.path.join(fig_folder + '\\' + 'qualityfirstqua.png')
plt.savefig(fig_filename)





#plot for each cluster the P2Pmax for pristine and pedot; Figure 2 I---------------------------------------------------

#cortex ketamine
#max_ampl_PEDOT = np.loadtxt(r'E:\Paper Impedance\bestP2PPEDOT_cortex_ket.txt')
max_ampl_PEDOT1 = np.loadtxt(r"C:\Users\KAMPFF-LAB-ANALYSIS3\Documents\Impedance paper\bestP2PPEDOT_cortex_ket.txt")
max_ampl_pristine1 = np.loadtxt(r"C:\Users\KAMPFF-LAB-ANALYSIS3\Documents\Impedance paper\bestP2Pristine_cortex_ket.txt")
fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
ax = plt.gca()
plt.scatter(max_ampl_pristine1, max_ampl_PEDOT1, color= 'k', label = 'Cortex/Ketamine')

#cortex urethane
max_ampl_PEDOT2 = np.loadtxt(r"C:\Users\KAMPFF-LAB-ANALYSIS3\Documents\Impedance paper\bestP2PPEDOT_cortex_uret.txt")
max_ampl_pristine2 = np.loadtxt(r"C:\Users\KAMPFF-LAB-ANALYSIS3\Documents\Impedance paper\bestP2Pristine_cortex_uret.txt")
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax = plt.gca()
plt.scatter(max_ampl_pristine2, max_ampl_PEDOT2, color ='r', label= 'Cortex/Urethane')

#hippocampus urethane
max_ampl_PEDOT3 = np.loadtxt(r"C:\Users\KAMPFF-LAB-ANALYSIS3\Documents\Impedance paper\bestP2PPEDOT_hippocampus_uret.txt")
max_ampl_pristine3 = np.loadtxt(r"C:\Users\KAMPFF-LAB-ANALYSIS3\Documents\Impedance paper\bestP2Pristine_hippocampus_uret.txt")
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax = plt.gca()
plt.scatter(max_ampl_pristine3, max_ampl_PEDOT3, color= '#2ca065', marker='s', label = 'Hippocampus/Urethane')

pristine= np.concatenate((max_ampl_pristine1,max_ampl_pristine2 , max_ampl_pristine3),axis=0)
pedot = np.concatenate((max_ampl_PEDOT1,max_ampl_PEDOT2 , max_ampl_PEDOT3),axis=0)

x = np.arange(800)
y = np.arange(800)
plt.plot(x, y, color='k', linestyle='dotted')
plt.xlabel('P2P Amplitude Pristine (\u00B5V)',fontsize=20)
plt.ylabel('P2P Amplitude PEDOT (\u00B5V)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(0, 800)
plt.xlim(0, 800)
plt.legend()

slope, intercept, r_value, p_value, std_err = stats.linregress(pristine,pedot)
print("r-squared:", r_value**2)
#plt.plot(pristine, pedot, 'o', label='original data')
plt.plot(pristine, intercept + slope*pristine, color='#5da4d6', label='fitted line')
plt.legend()
plt.show()


#______________________________________________________________________________________________________________________


#nice colors codes hexa
#2ca065 (green)
#ff4136(red)
#cf72ff(purple)
#5da4d6 (blue)
#ff900e (orange)
#'#f3f3f3' grey






























#code not used---------------------------------------------------------------------------------------------------------

#Code from steve to plot cluster quality

def plot_cluster_quality_by_group(cluster_groups, sorting_quality_parameter):
    """
    visualise any sorting quality parameter according to the classification clusters of each group
    groups 0:
    :param cluster_groups:
    :param sorting_quality_parameter:
    :return:
    """

    fig = plt.figure()
    for group in np.unique(cluster_groups)[0:3]:
        in_this_group = cluster_groups == group
        these_unit_qualities = sorting_quality_parameter[in_this_group]
        group_mean = np.mean(these_unit_qualities)
        plt.scatter(np.ones_like(these_unit_qualities)*group, these_unit_qualities, color='k', alpha=0.5)
        plt.scatter(group, group_mean, s=50)
    return fig


plot_cluster_quality_by_group(cluster_groups, contamination_rate)
plot_cluster_quality_by_group(cluster_groups, unit_quality)
plot_cluster_quality_by_group(cluster_groups, isi_violation)



def quality_box_plot(df):
    y_params = ['unit_quality', 'contamination_rate', 'isi_violations']
    fig = plt.figure()

    for i, y_param in enumerate(y_params):
        fig.add_subplot(2, 2, i+1)
        sns.set(style="ticks", palette="deep", color_codes=True)

        sns.boxplot(x="group", y=y_param, data=df,
                    whis=np.inf, color="c")

        # Add in points to show each observation
        sns.stripplot(x="group", y=y_param, data=df,
                      jitter=True, size=3, color=".3", linewidth=0)


group_dict_for_quality = {
    'noise': 0,
    'MUA': 1,
    'good': 2,
    'unsorted': 3}




# create folder to save figures ---------------------------------------------------------------------------------------
fig_folder = os.path.join(analysis_folder+ '\\'+ 'figures' +  '\\' + 'amplifier' + date + 'T' +
  all_recordings_capture_times[surgery][pick_recording])

if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)


#plot best channel PEDOT versus Pristine

voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000

for pick_cluster in clusterSelected:
    f = plt.figure()
    ax = plt.gca()
    filename_p2p = os.path.join(analysis_folder + '\\' + 'p2p_EXTRA_Cluster' + date + '_' + pick_recording +  '_' + str(pick_cluster) + '.npy')
    amplitudes = np.load(filename_p2p)
    select_indices_biggestAmp = np.array(np.where(amplitudes >= (amplitudes.max()/2)))

    if select_indices_biggestAmp.shape[1] <= 1:
        print('cluster ' + str(pick_cluster) + 'NOT GOOD')

    else:
        channel = amplitudes.argmax()
        select_pristine = np.intersect1d(select_indices_biggestAmp, pristine_all)
        select_pedot = np.intersect1d(select_indices_biggestAmp, pedot_all)
        best_channel_pedot = select_pedot[amplitudes[select_pedot].argmax()]
        best_channel_pridtine = select_pristine[amplitudes[select_pristine].argmax()]
        print('cluster' + str(pick_cluster))
        print('pedot channel' + str(best_channel_pedot))
        print('pristine channel' + str(best_channel_pridtine))
        num_samples = all_cells_ivm_filtered_data[pick_cluster][:,:,:].shape[1]
        sample_axis = np.arange(-(num_samples/2),(num_samples/2))
        time_axis = sample_axis/sampling_freq
        num_of_spikes = len(kilosort_units[pick_cluster])
        subset = np.random.choice(np.arange(num_of_spikes), math.ceil(num_of_spikes * 1))
        extra_average_Pristine = np.average(all_cells_ivm_filtered_data[pick_cluster][best_channel_pridtine,:,:],axis=1) * voltage_step_size * scale_uV
        extra_average_PEDOT = np.average(all_cells_ivm_filtered_data[pick_cluster][best_channel_pedot,:,:],axis=1) * voltage_step_size * scale_uV

        #pristine best channel
        ax.plot(time_axis * scale_ms, all_cells_ivm_filtered_data[pick_cluster][best_channel_pridtine,:,subset].T * voltage_step_size * scale_uV, color='#5da4d6', alpha=0.002)
        ax.set_title('cluster'+str(pick_cluster))
        ax.plot(time_axis * scale_ms, extra_average_Pristine, color='#5da4d6', linewidth=3)
        #pedot best channel
        ax.plot(time_axis * scale_ms, all_cells_ivm_filtered_data[pick_cluster][best_channel_pedot,:,subset].T * voltage_step_size * scale_uV, color='#ff4136', alpha = 0.005)
        ax.plot(time_axis * scale_ms, extra_average_PEDOT, color='#ff4136', linewidth=3, label='%i'% num_of_spikes)
        plt.xlim(-1, 1)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.xlabel('Time (ms)',fontsize=20)
    plt.ylabel('Voltage (\u00B5V)', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    fig_filename = os.path.join(fig_folder, types_of_sorting[cluster_to_load].format(pick_cluster) + 'VoltagePEDOTvsPristine.png')
    plt.savefig(fig_filename)
    plt.legend()
    plt.close()


#plot P2P max for pristine and pedot

best_channel_pedot = np.loadtxt(r'E:\Paper Impedance\bestPEDOTchannel.txt')
best_channel_pristine = np.loadtxt(r'E:\Paper Impedance\bestPristinechannel.txt')
max_ampl_PEDOT = np.loadtxt(r'E:\Paper Impedance\bestP2PPEDOT_1.txt')
max_ampl_pristine = np.loadtxt(r'E:\Paper Impedance\bestP2PPristine_2.txt')


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(best_channel_pedot, max_ampl_PEDOT, color= 'r')
xt = pedot_all
ax.set_xticklabels(xt)
ax.scatter( best_channel_pristine,max_ampl_pristine, color = 'b')
yt = pristine_all
ax.set_yticklabels(yt)



#plot P2P min for pristine and pedot-----------------------------------------------------------------------------------

#plot cortex ketam

min_ampl_PEDOT = np.loadtxt(r"Z:\j\Joana Neto\Backup_2017_28_06\PCDisk\Paper Impedance\Figures\plot_2P2_pristine_pedot\min_bestP2PPEDOT_cortex_ket.txt")
min_ampl_pristine = np.loadtxt(r"Z:\j\Joana Neto\Backup_2017_28_06\PCDisk\Paper Impedance\Figures\plot_2P2_pristine_pedot\min_bestP2Pristine_cortex_ket.txt")

fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
ax = plt.gca()
plt.scatter(min_ampl_pristine, min_ampl_PEDOT, color= 'k', label = 'Cortex/Ketamine')



#plot cortex uret
min_ampl_PEDOT = np.loadtxt(r"Z:\j\Joana Neto\Backup_2017_28_06\PCDisk\Paper Impedance\Figures\plot_2P2_pristine_pedot\min_bestP2PPEDOT_cortex_uret.txt")
min_ampl_pristine = np.loadtxt(r"Z:\j\Joana Neto\Backup_2017_28_06\PCDisk\Paper Impedance\Figures\plot_2P2_pristine_pedot\min_bestP2Pristine_cortex_uret.txt")

#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax = plt.gca()
plt.scatter(min_ampl_pristine, min_ampl_PEDOT, color ='r', label= 'Cortex/Urethane')



#plot hippocamp uret

min_ampl_PEDOT = np.loadtxt(r"Z:\j\Joana Neto\Backup_2017_28_06\PCDisk\Paper Impedance\Figures\plot_2P2_pristine_pedot\min_bestP2PPEDOT_hippocampus_uret.txt")
min_ampl_pristine = np.loadtxt(r"Z:\j\Joana Neto\Backup_2017_28_06\PCDisk\Paper Impedance\Figures\plot_2P2_pristine_pedot\min_bestP2Pristine_hippocampus_uret.txt")

#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax = plt.gca()
plt.scatter(min_ampl_pristine, min_ampl_PEDOT, color= '#cf72ff', label = 'Hippocampus/Urethane')
x = np.arange(800)
y = np.arange(800)
plt.plot(x, y, color='k')
plt.xlabel('P2P Voltage Pristine (\u00B5V)',fontsize=20)
plt.ylabel('P2P Voltage PEDOT (\u00B5V)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(0, 800)
plt.xlim(0, 800)
plt.legend()



def _generate_adjacency_graph(all_electrodes):
    graph_dict = {}
    for r in np.arange(all_electrodes.shape[0]):
        for c in np.arange(all_electrodes.shape[1]):
            electrode = all_electrodes[r, c]
            if electrode != -1:
                graph_dict[electrode] = []
                for step_r in np.arange(-1, 2):
                    if -1 < r + step_r < all_electrodes.shape[0]:
                        for step_c in np.arange(-1, 2):
                            if -1 < c + step_c < all_electrodes.shape[1] and not (step_r ==0 and step_c == 0):
                                neighbour = all_electrodes[r + step_r, c + step_c]
                                if neighbour != -1:
                                    graph_dict[electrode].append(all_electrodes[r + step_r, c + step_c])
                if len(graph_dict[electrode]) == 0:
                    graph_dict.pop(electrode)

    return graph_dict
