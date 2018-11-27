import itertools
import os
import warnings

import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import scipy.signal as signal
import scipy.stats as stats

import IO.ephys as ephys
# import Utilities as ut

import matplotlib.animation as animation

plt.rcParams['animation.ffmpeg_path'] ="C:\\Users\\KAMPFF-LAB_ANALYSIS4\\Downloads\\ffmpeg-20160116-git-d7c75a5-win64-static\\ffmpeg-20160116-git-d7c75a5-win64-static\\bin\\ffmpeg.exe"


base_folder = r'Z:\j\Neuroseeker256ch'

date = {1: '2017-02-08', 2:''}

all_recordings_capture_times = {1: {'1': '21_38_55', '2': '20_04_54', '3': '15_34_04'},
                                2: {'1':'','2':''}}

recordings_cluster = {1: ['1','2','3'],
                      2:['1','2']}

cluster_indices_good = {1: {'1': [46, 186, 252, 202], '2': [61, 2, 27, 23, 42, 54], '3': [23, 19, 37, 83, 91, 49, 40, 78, 85, 49, 52, 9, 5]},
                        2:{'1': [], '2': []}}

#nfilt2048= [48, 38,26,77,102,61,15,106,60]
#nfilt1024= [75,63,35,13,71,24,93,44]
#nfilt512= [23, 19, 37, 83, 91, 49, 40, 78, 85, 49, 52, 9, 5]
#----------------------------------------------------------------------------------------------------------------------

surgery = 1
pick_recording = '3' #we need to select the recording


recordings = recordings_cluster[surgery]
clusterSelected = cluster_indices_good[surgery][pick_recording]


date = date[surgery]
data_folder = os.path.join(base_folder + '\\' + 'Type1_Probe6'+'\\'+ date, 'Data')
analysis_folder = os.path.join(base_folder + '\\' +'Type1_Probe6'+'\\'+ date, 'Analysis')

recordings_capture_times = all_recordings_capture_times[surgery]

#----------------------------------------------------------------------------------------------------------------------

num_of_points_in_spike_trig_ivm = 128
num_of_points_for_padding = 2 * num_of_points_in_spike_trig_ivm


inter_spike_time_distance = 30

num_ivm_channels = 256
amp_dtype = np.uint16

sampling_freq = 20000
high_pass_freq = 100
filtered_data_type = np.float64

types_of_data_to_load = {'t': 'ivm_data_filtered_cell{}.dat',
                         'f': 'ivm_data_filtered_cell{}',
                         'c': 'ivm_data_filtered_continous_cell{}.dat',
                         'k': 'ivm_data_filtered_klusta_spikes_cell{}.dat',
                         'm': 'ivm_data_raw_cell{}.dat',
                         }

types_of_sorting = {'s': 'goodcluster_{}_'}


#Filter for extracellular recording
def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=20000.0,passFreq=100.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)

#kilosort units

#nfilt = 'nfilt2048'
#nfilt = 'nfilt1024'
nfilt = 'nfilt512'

spike_template_folder = os.path.join(base_folder + '\\' +'Type1_Probe6'+'\\'+ date + '\\' + 'Datakilosort' +  '\\' + 'Nfilt_Test' + '\\' + nfilt + '\\' + 'amplifier' + date + 'T' + all_recordings_capture_times[surgery][pick_recording])
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

    return good_clusters,n_good_clusters,n_mua_clusters,n_noise_clusters


[good_clusters,n_good_clusters,n_mua_clusters,n_noise_clusters]= find_good_clusters(spike_template_folder)
total = n_good_clusters +n_mua_clusters+n_noise_clusters

# Generate the (channels x time_points x spikes) high passed extracellular recordings datasets for all cells
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


# Load the extracellular recording cut data from the .dat files on hard disk onto memmaped arrays

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


lag = 50

for pick_cluster in clusterSelected:
    times_cluster = np.reshape(spike_times[kilosort_units[pick_cluster]], (spike_times[kilosort_units[pick_cluster]].shape[0]))
    diffs, norm = crosscorrelate_spike_trains((times_cluster/30).astype(np.int64),(times_cluster/30).astype(np.int64),lag=50)
    #hist, edges = np.histogram(diffs)
    plt.figure()
    plt.hist(diffs, bins=101, normed=False, range=(-50,50), align ='mid') #if normed = True, the probability density function at the bin, normalized such that the integral over the range is 1.
    plt.xlim(-lag,lag)


#Crosscocorrelagram between similar cluster--------------------------------------------------------------------------------------


def crosscorrelate(sua1, sua2, lag=None, n_pred=1, predictor=None,
                   display=False, kwargs={}):

    assert predictor is 'shuffle' or predictor is None, "predictor must be \
    either None or 'shuffle'. Other predictors are not yet implemented."
    #Check whether sua1 and sua2 are SpikeTrains or arrays
    sua = []
    for x in (sua1, sua2):
        #if isinstance(x, SpikeTrain):
        if hasattr(x, 'spike_times'):
            sua.append(x.spike_times)
        elif x.ndim == 1:
            sua.append(x)
        elif x.ndim == 2 and (x.shape[0] == 1 or x.shape[1] == 1):
            sua.append(x.ravel())
        else:
            raise TypeError("sua1 and sua2 must be either instances of the" \
                            "SpikeTrain class or column/row vectors")
    sua1 = sua[0]
    sua2 = sua[1]
    if sua1.size < sua2.size:
        if lag is None:
            lag = np.ceil(10*np.mean(np.diff(sua1)))
        reverse = False
    else:
        if lag is None:
            lag = np.ceil(20*np.mean(np.diff(sua2)))
        sua1, sua2 = sua2, sua1
        reverse = True
    #construct predictor
    if predictor is 'shuffle':
        isi = np.diff(sua2)
        sua2_ = np.array([])
        for ni in xrange(1,n_pred+1):
            idx = np.random.permutation(isi.size-1)
            sua2_ = np.append(sua2_, np.add(np.insert(
                (np.cumsum(isi[idx])), 0, 0), sua2.min() + (
                np.random.exponential(isi.mean()))))
    #calculate cross differences in spike times
    differences = np.array([])
    pred = np.array([])
    for k in np.arange(0, sua1.size): #changed xrange() for np.arange()
        differences = np.append(differences, sua1[k] - sua2[np.nonzero(
            (sua2 > sua1[k] - lag) & (sua2 < sua1[k] + lag))])
    if predictor == 'shuffle':
        for k in np.arange(0, sua1.size): #changed xrange() for np.arange()
            pred = np.append(pred, sua1[k] - sua2_[np.nonzero(
                (sua2_ > sua1[k] - lag) & (sua2_ < sua1[k] + lag))])
    if reverse is True:
        differences = -differences
        pred = -pred
    norm = np.sqrt(sua1.size * sua2.size)
    return differences, pred, norm


lag = 50
pick_cluster1 = 48
pick_cluster2 = 60
#pick_cluster3 = 52
times_cluster1 = np.reshape(spike_times[kilosort_units[pick_cluster1]], (spike_times[kilosort_units[pick_cluster1]].shape[0]))
times_cluster2 = np.reshape(spike_times[kilosort_units[pick_cluster2]], (spike_times[kilosort_units[pick_cluster2]].shape[0]))
#times_cluster3 = np.reshape(spike_times[kilosort_units[pick_cluster3]], (spike_times[kilosort_units[pick_cluster3]].shape[0]))

diffs, pred, norm = crosscorrelate((times_cluster1/30).astype(np.int64),(times_cluster2/30).astype(np.int64),lag=50)
#hist, edges = np.histogram(diffs)
plt.figure()
plt.hist(diffs, bins=101, normed=False, range=(-50,50), align ='mid') #if normed = True, the probability density function at the bin, normalized such that the integral over the range is 1.
plt.xlim(-lag,lag)

# MAX of detections for each channel
max=[]
index=[]
diffs, pred, norm = crosscorrelate((all_spikes_2048/30).astype(np.int64),(all_spikes_1024/30).astype(np.int64),lag=50)
plt.figure()
n,b,p = plt.hist(diffs,bins=101, range=(-50,50), align='mid')
max.append(n.max())
index.append(n.argmax())

max = np.array(max)
index = np.array(index)

print(max.max())
print(max.argmax())



#plotspike times

plt.figure()
plt.vlines(times_cluster1, ymin = 0,ymax = 1, label = np.str(pick_cluster1))
plt.vlines(times_cluster2, ymin=1,ymax=2,color='m',label = np.str(pick_cluster2))
plt.vlines(times_cluster3, ymin=2,ymax=3,color='r',label = np.str(pick_cluster3))
plt.ylim([0,5])

plt.legend()





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

def triggerline(x):
    if x is not None:
        ylim = plt.ylim()
        plt.vlines(x,ylim[0],ylim[1])


# Return mapping to plot in space---------------------------------------------------------------------------------------
def create_256channels_neuroseeker_prb(bad_channels=None):
    '''
     This function produces a grid with the electrodes positions for the 256 channel probe

     Inputs:
     bad_channels is a list or numpy array with channels you may want to
     disregard.

     Outputs:
     channel_positions is a Pandas Series with the electrodes positions (in
     two dimensions)
     '''


    electrode_amplifier_index_on_grid = np.genfromtxt(r'Z:\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\2017-02-08\chanmap.csv', delimiter=",")
    all_electrodes = electrode_amplifier_index_on_grid.astype(np.int16)

    return all_electrodes



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


#test distance

def _generate_adjacency_graph(all_electrodes):

    graph_dict = {}
    distance = {}
    for r in np.arange(all_electrodes.shape[0]):
        for c in np.arange(all_electrodes.shape[1]):
            electrode = all_electrodes[r, c]
            if electrode != -1:
                graph_dict[electrode] = []
                distance[electrode] = []
                for step_r in np.arange(-all_electrodes.shape[0], all_electrodes.shape[0]):
                    if -1 < r + step_r < all_electrodes.shape[0]:
                        for step_c in np.arange(-all_electrodes.shape[1], all_electrodes.shape[1]):
                            if -1 < c + step_c < all_electrodes.shape[1] and not (step_r ==0 and step_c == 0):
                                neighbour = all_electrodes[r + step_r, c + step_c]
                                if neighbour != -1:
                                    graph_dict[electrode].append(all_electrodes[r + step_r, c + step_c])
                                    distance[electrode].append(np.sqrt((r - r + step_r)**2 + (c-c + step_c)**2))
                if len(graph_dict[electrode]) == 0:
                    graph_dict.pop(electrode)
                if len(distance[electrode]) == 0:
                    distance.pop(electrode)
    return graph_dict, distance

#test2

def _generate_adjacency_graph2(all_electrodes):
    graph_dict = {}
    distance = {}
    for r in np.arange(all_electrodes.shape[0]):
        for c in np.arange(all_electrodes.shape[1]):
            electrode = all_electrodes[r, c]
            if electrode != -1:
                graph_dict[electrode] = []
                distance[electrode] = []
                for step_r in np.arange(-all_electrodes.shape[0], all_electrodes.shape[0]):
                    if -1 < r + step_r < all_electrodes.shape[0]:
                        for step_c in np.arange(-all_electrodes.shape[1], all_electrodes.shape[1]):
                            if -1 < c + step_c < all_electrodes.shape[1] and not (step_r ==0 and step_c == 0):
                                neighbour = all_electrodes[r + step_r, c + step_c]
                                if neighbour != -1:
                                    try:
                                        graph_dict[neighbour]
                                    except:
                                        graph_dict[electrode].append(all_electrodes[r + step_r, c + step_c])
                                        distance[electrode].append(np.sqrt((r - r + step_r)**2 + (c-c + step_c)**2))

                                    else:
                                        try:
                                            graph_dict[neighbour].index(electrode)
                                        except:
                                            graph_dict[electrode].append(all_electrodes[r + step_r, c + step_c])
                                            distance[electrode].append(np.sqrt((r - r + step_r)**2 + (c-c + step_c)**2))

                if len(graph_dict[electrode]) == 0:
                    graph_dict.pop(electrode)
                if len(distance[electrode]) == 0:
                    distance.pop(electrode)
    return graph_dict, distance









electrodes_highZ = []
all_electrodes = create_256channels_neuroseeker_prb(bad_channels=[-1])
graph_dict = _generate_adjacency_graph(all_electrodes)

# Plot 256 channels averages overlaid
all_electrodes = create_256channels_neuroseeker_prb()
voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000

def plot_average_extra(all_cells_ivm_filtered_data, electrode_structure=all_electrodes, yoffset=1):
    origin_cmap= plt.get_cmap('hsv')
    shrunk_cmap = shiftedColorMap(origin_cmap, start=0.6, midpoint=0.7, stop=1, name='shrunk')
    cm = shrunk_cmap
    cNorm = colors.Normalize(vmin=0,vmax=255)
    scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cm)
    subplot_number_array = electrode_structure.reshape(1,255)
    for clus in np.arange(0, len(clusterSelected)):
        num_of_spikes = len(kilosort_units[clusterSelected[clus]])
        extra_average_V = np.average(all_cells_ivm_filtered_data[clusterSelected[clus]][:,:,:],axis=2)
        num_samples=extra_average_V.shape[1]
        sample_axis= np.arange(-(num_samples/2),(num_samples/2))
        time_axis= sample_axis/sampling_freq
        extra_average_microVolts = extra_average_V * scale_uV * voltage_step_size
        plt.figure()
        for m in np.arange(np.shape(extra_average_microVolts)[0]-1):
            colorVal = scalarMap.to_rgba(np.shape(extra_average_microVolts)[0]-m)
            plt.plot(time_axis * scale_ms, extra_average_microVolts[subplot_number_array[:,m],:].T, color=colorVal)
            plt.xlim(-2, 2) #window 4ms
            plt.ylim(np.min(extra_average_microVolts)-yoffset,np.max(extra_average_microVolts)+yoffset )
            plt.ylabel('Voltage (\u00B5V)', fontsize=20)
            plt.xlabel('Time (ms)',fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            triggerline(0)


# Plot 256 channels averages in space
all_electrodes = create_256channels_neuroseeker_prb()
voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000

def plot_average_extra_geometry(all_cells_ivm_filtered_data, electrode_structure=all_electrodes):
    origin_cmap= plt.get_cmap('hsv')
    shrunk_cmap = shiftedColorMap(origin_cmap, start=0.6, midpoint=0.7, stop=1, name='shrunk')
    cm=shrunk_cmap
    cNorm=colors.Normalize(vmin=0,vmax=255)
    scalarMap= cmx.ScalarMappable(norm=cNorm,cmap=cm)
    subplot_number_array = electrode_structure.reshape(1,255)
    for clus in np.arange(0, len(clusterSelected)):
        plt.figure(clus+1)
        num_of_spikes = len(kilosort_units[clusterSelected[clus]])
        extra_average_V = np.average(all_cells_ivm_filtered_data[clusterSelected[clus]][:,:,:],axis=2)
        num_samples=extra_average_V.shape[1]
        sample_axis= np.arange(-(num_samples/2),(num_samples/2))
        time_axis= sample_axis/sampling_freq
        extra_average_microVolts = extra_average_V * scale_uV * voltage_step_size
        for i in np.arange(1, subplot_number_array.shape[1]+1):
            plt.subplot(17,15,i)
            colorVal = scalarMap.to_rgba(subplot_number_array.shape[1]-i)
            plt.plot(time_axis*scale_ms, extra_average_microVolts[np.int(subplot_number_array[:,i-1]),:],color=colorVal)
            plt.ylim(np.min(np.min(extra_average_microVolts, axis=1)), np.max(np.max(extra_average_microVolts, axis=1)))
            plt.xlim(-2, 2)
            plt.axis("OFF")

#--------------------------------------------------------------------------------------------------------------------
# Each cluster P2P, MIN, MAX 256channels

voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000
#voltage_step_size = 1
#scale_uV = 1


def peaktopeak(all_cells_ivm_filtered_data, windowSize=60):

    for clus in np.arange(0, len(clusterSelected)):
        extra_average_V= np.average(all_cells_ivm_filtered_data[clusterSelected[clus]][:,:,:],axis=2) * voltage_step_size
        NumSamples=extra_average_V.shape[1]
        extra_average_microVolts = extra_average_V * scale_uV
        NumSites=np.size(extra_average_microVolts,axis = 0)
        lowerBound=int(NumSamples/2.0-windowSize/2.0)
        upperBound=int(NumSamples/2.0+windowSize/2.0)

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
           stdv_minimos[b]= stdv[b, argminima[b]]
           stdv_maximos[b]= stdv[b, argmaxima[b]]

        error =  np.sqrt((stdv_minimos * stdv_minimos)+ (stdv_maximos*stdv_maximos))


        np.save(os.path.join(analysis_folder,'stdv_minimos_EXTRA_Cluster'+ str(clusterSelected[clus]) + '.npy'), stdv_minimos)
        np.save(os.path.join(analysis_folder,'stdv_maximos_EXTRA_Cluster'+ str(clusterSelected[clus])  + '.npy'), stdv_maximos)
        np.save(os.path.join(analysis_folder,'stdv_average_EXTRA_Cluster'+ str(clusterSelected[clus])+ '.npy'), stdv)
        np.save(os.path.join(analysis_folder,'error_EXTRA_Cluster'+ str(clusterSelected[clus])+ '.npy'), error)
        np.save(os.path.join(analysis_folder,'p2p_EXTRA_Cluster'+ str(clusterSelected[clus]) + '.npy'), p2p)
        np.save(os.path.join(analysis_folder,'minima_EXTRA_Cluster'+ str(clusterSelected[clus]) + '.npy'), minima)
        np.save(os.path.join(analysis_folder,'maxima_EXTRA_Cluster'+ str(clusterSelected[clus])+ '.npy'), maxima)
        np.save(os.path.join(analysis_folder,'argmaxima_EXTRA_Cluster'+ str(clusterSelected[clus]) + '.npy'), argmaxima)
        np.save(os.path.join(analysis_folder,'argminima_EXTRA_Cluster'+ str(clusterSelected[clus]) + '.npy'), argminima)

    return argmaxima, argminima, maxima, minima, p2p

#Print the P2P. MIN, MAX------------------------------------------------------------------------------------------------

for pick_cluster in clusterSelected:
    filename_p2p = os.path.join(analysis_folder + '\p2p_EXTRA_Cluster' + str(pick_cluster) + '.npy')
    filename_error= os.path.join(analysis_folder +  '\error_EXTRA_Cluster' + str(pick_cluster) +'.npy')
    amplitudes = np.load(filename_p2p)
    error = np.load(filename_error)
    min_ampl= amplitudes.min()
    max_ampl= amplitudes.max()
    channel = amplitudes.argmax()

    print('#------------------------------------------------------')
    print('recording_'+ str(pick_recording))
    print('cluster_'+ str(pick_cluster))
    print('min_p2p ' + str(min_ampl))
    print('max_p2p ' + str(max_ampl))
    print('maxp2p_channel ' + str(channel))
    select_indices_biggestAmp = np.where(amplitudes >= (amplitudes.max()/2))
    print('Channels_atleasthalfofmaxp2p ' + str(select_indices_biggestAmp))
    Amp_select_indices = amplitudes[select_indices_biggestAmp]
    print('Amplitude_atleasthalfofmaxp2p ' + str(Amp_select_indices))
    print('#------------------------------------------------------')

#Plot Heatmap and movies------------------------------------------------------------------------------------------------

def polytrode_256channels(bad_channels=[]):
    '''
     This function produces a grid with the electrodes positions for the 256 channel probe

     Inputs:
     bad_channels is a list or numpy array with channels you may want to
     disregard.

     Outputs:
     channel_positions is a Pandas Series with the electrodes positions (in
     two dimensions)
     '''

    electrode_coordinate_grid = list(itertools.product(np.arange(0, 18),
                                                       np.arange(0, 15)))

    electrode_amplifier_index_on_grid = np.genfromtxt(r'Z:\j\Neuroseeker256ch_probe6\2017-02-08\chanmap256.csv', delimiter=",")

    electrode_amplifier_index_on_grid = electrode_amplifier_index_on_grid.astype(np.int16)


    reshaped = np.reshape(electrode_amplifier_index_on_grid,np.shape(electrode_amplifier_index_on_grid)[0]*np.shape(electrode_amplifier_index_on_grid)[1])

    electrode_amplifier_index_on_grid = reshaped
    electrode_amplifier_name_on_grid = np.array(["Int"+str(x) for x in electrode_amplifier_index_on_grid])
    indices_arrays = [electrode_amplifier_index_on_grid.tolist(), electrode_amplifier_name_on_grid.tolist()]
    indices_tuples = list(zip(*indices_arrays))

    channel_position_indices = pd.MultiIndex.from_tuples(indices_tuples, names=['Numbers', 'Strings'])
    channel_positions = pd.Series(electrode_coordinate_grid, index = channel_position_indices)

    channel_positions.columns=['Positions']
    channel_positions = channel_positions.reset_index(level=None, drop=False, name='Positions')
    if bad_channels is not None:
        channel_positions = channel_positions[~channel_positions.Numbers.isin(bad_channels)]
    return channel_positions


def plot_topoplot256(axis, channel_positions, data, show=True, **kwargs):
    '''
    This function interpolates the data between electrodes and plots it into
    the output.

    Inputs:
    axis is an instance of matplotlib.axes where you want the heatmap to be
    output.
    channel_positions is a Pandas Series with the positions of the electrodes
    (this is the output of polytrode_channels function)
    data is a numpy array containing the data to be interpolate and then
    displayed.
    show is a boolean variable to assert whether you want the heatmap to be
    printed on the screen
    kwargs can :
    - hpos and vpos define the horizontal and vertical position offset of the
      output heatmap, respectively.
    - width and height define the horizontal and vertical scale of the output
      heatmap, respectively.
    - gridscale defines the resolution of the interpolation.
    - interpolation_method defines the method used to interpolate the data
      between positions in channel_positions.
      Choose from:
      ‘none’, ‘nearest’, ‘bilinear’, ‘bicubic’, ‘spline16’, ‘spline36’,
      ‘hanning’, ‘hamming’, ‘hermite’, ‘kaiser’, ‘quadric’, ‘catrom’,
      ‘gaussian’, ‘bessel’, ‘mitchell’, ‘sinc’, ‘lanczos’
    - zlimits defines the limits of the amplitude of the output heatmap.

    Outputs:
    image is the heatmap.
    scat is the grid of electrodes.
    '''
    if not kwargs.get('hpos'):
        hpos = 0
    else:
        hpos = kwargs['hpos']
    if not kwargs.get('vpos'):
        vpos = 0
    else:
        vpos = kwargs['vpos']
    if not kwargs.get('width'):
        width = None
    else:
        width = kwargs['width']
    if not kwargs.get('height'):
        height = None
    else:
        height = kwargs['height']
    if not kwargs.get('gridscale'):
        gridscale = 1
    else:
        gridscale = kwargs['gridscale']
    if not kwargs.get('interpolation_method'):
        interpolation_method = "bicubic"
    else:
        interpolation_method = kwargs['interpolation_method']
    if not kwargs.get('zlimits'):
        zlimits = None
    else:
        zlimits = kwargs['zlimits']

    if np.isnan(data).any():
        warnings.warn('The data passed to gdft_plot_topo contain NaN values. \
        These will create unexpected results in the interpolation. \
        Deal with them.')

    channel_positions = channel_positions.sort('Numbers', ascending=True)
    channel_positions = np.array([[x, y] for x, y in channel_positions.Positions])
    allCoordinates = channel_positions

    naturalWidth = np.max(allCoordinates[:, 0]) - np.min(allCoordinates[:, 0])
    naturalHeight = np.max(allCoordinates[:, 1]) - np.min(allCoordinates[:, 1])

    if not width and not height:
        xScaling = 1
        yScaling = 1
    elif not width and height:
        yScaling = height/naturalHeight
        xScaling = yScaling
    elif width and not height:
        xScaling = width/naturalWidth
        yScaling = xScaling
    elif width and height:
        xScaling = width/naturalWidth
        yScaling = height/naturalHeight

    chanX = channel_positions[:, 0] * xScaling + hpos
    chanY = channel_positions[:, 1] * yScaling + vpos
    chanX = np.max(chanX) - chanX

    hlim = [np.min(chanY), np.max(chanY)]
    vlim = [np.min(chanX), np.max(chanX)]

    if interpolation_method is not 'none':
        yi, xi = np.mgrid[hlim[0]:hlim[1]:complex(0, gridscale)*(hlim[1]-hlim[0]), vlim[0]:vlim[1]:complex(0, gridscale)*(vlim[1]-vlim[0])]
    else:
        yi, xi = np.mgrid[hlim[0]:hlim[1]+1, vlim[0]:vlim[1]+1]         # for no interpolation show one pixel per data point

    Zi = interpolate.griddata((chanX, chanY), data, (xi, yi))

    if not zlimits:
        vmin = data.min()
        vmax = data.max()
    else:
        vmin = zlimits[0]
        vmax = zlimits[1]

    cmap = plt.get_cmap("jet")
    image = axis.imshow(Zi.T, cmap=cmap, origin=['lower'], vmin=vmin,
                        vmax=vmax, interpolation=interpolation_method,
                        extent=[hlim[0], hlim[1], vlim[0], vlim[1]],
                        aspect='equal')

    scat = axis.scatter(chanY, chanX)
    if show:
        plt.colorbar(image)
        plt.show()
    return image, scat

#Method1 by using plot_topoplot

for pick_cluster in clusterSelected:
    filename_p2p = os.path.join(analysis_folder + '\p2p_EXTRA_Cluster' + str(pick_cluster) + '.npy')
    filename_error= os.path.join(analysis_folder +  '\error_EXTRA_Cluster' + str(pick_cluster) +'.npy')
    amplitudes = np.load(filename_p2p)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    channel_positions=polytrode_256channels(bad_channels=[-1])
    image, scat = plot_topoplot256(ax1, channel_positions, amplitudes, show=False, interpmethod="quadric", gridscale=5, zlimits=[np.min(amplitudes), np.max(amplitudes)])
    ax1.set_yticks([], minor=False)
    ax1.set_xticks([], minor=False)
    plt.colorbar(mappable=image, ticks = [np.min(amplitudes), 0, np.max(amplitudes)])
    plt.show()




def plot_video_topoplot_with_juxta256(data, time_axis, channel_positions,
                        times_to_plot=[5.4355, 5.439], time_window=0.000060,
                        time_step=0.000060, sampling_freq=20000,
                        zlimits=[-300, 100],filename=r'Z:\j\Neuroseeker256ch_probe6\2017-02-08\Analysis\\mymovie.avi'):
    '''
    This function creates a video file of time ordered heatmaps.

    Inputs:
    data is a numpy array containing the data to produce all the heatmaps.
    time_axis is a numpy array containing the time points of sampling. (in
    seconds)
    channel_positions is a Pandas Series with the positions of the electrodes
    (this is the output of polytrode_channels function)
    times_to_plot defines the upper and lower bound of the time window to be
    used. (in seconds)
    time_window defines the size of the window to calculate the average. (in
    seconds)
    time_step is the time interval between consecutive heatmaps (in seconds).
    sampling_freq is the sampling frequency, in Hz, at which the data was
    acquired.
    zlimits defines the limits of the magnitude of all heatmaps.
    filename is the path to the output video file.
    '''
    global images, sub_time_indices

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    #ax2 = fig.add_subplot(1, 2, 2)
    sample_step = int(time_step * sampling_freq)
    sub_time_indices = np.arange(find_closest(time_axis, times_to_plot[0]),
                                 find_closest(time_axis, times_to_plot[1]))
    sub_time_indices = sub_time_indices[0::sample_step]
    print(sub_time_indices)
    text_y = 17
    text_x = -8
    images = []
    for t in sub_time_indices:
        samples = [t, t + (time_window*sampling_freq)]
        print(t)
        data_to_plot = np.mean(data[:, int(samples[0]):int(samples[1])], 1)

        image, scat = plot_topoplot256(ax1, channel_positions, data_to_plot,
                                    show=False, interpmethod="quadric",
                                    gridscale=5, zlimits=zlimits)
        ax1.set_yticks([], minor=False)
        ax1.set_xticks([], minor=False)
        stringy = '%.2f' % (time_axis[t]*1000)
        stringy = 't' + '=' + stringy + ' ms'
        txt = plt.text(x=text_x, y=text_y, s=stringy)
        #ax2.text(4,22, '\u00B5V')
        ax1.set_title("Voltage at t / \u00B5V")

        #if t < np.max(sub_time_indices) - 20 - np.min(sub_time_indices):
        #    pointsToPlot = juxtaData[ t-10:t-10+20]
        #else:
        #    pointsToPlot = juxtaData[t-10:t-10+20]

        #grafico, = ax1.plot(np.arange(-10, 10)/20., pointsToPlot, 'b')
        #ax1.set_yticks([0], minor=False)
        #ax1.set_yticks([0], minor=True)
        #ax1.yaxis.grid(False, which='major')
        #ax1.yaxis.grid(False, which='minor')
        #ax1.set_xticks([0])
        #ax1.grid(True)
        #ax1.set_title("Juxta Signal/ mV")
        images.append([image, scat, txt])
    FFwriter = animation.FFMpegWriter()
    ani = animation.ArtistAnimation(fig, images, interval=1000, blit=True,repeat_delay=1000)
    plt.colorbar(mappable=image)

    if filename is not None:
        ani.save(filename, writer=FFwriter, fps=0.5, bitrate=5000, dpi=300, extra_args=['h264'])
    plt.show()

def find_closest(array, target):
    # a must be sorted
    idx = array.searchsorted(target)
    idx = np.clip(idx, 1, len(array)-1)
    left = array[idx-1]
    right = array[idx]
    idx -= target - left < right - target
    return idx

#Method by using plot_video_topoplot_with_juxta256

voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_mV = 1000
sampling_freq = 20000
channel_positions=polytrode_256channels(bad_channels=[-1])

for pick_cluster in clusterSelected:
    rootDir = r'Z:\j\Neuroseeker256ch_probe6\2017-02-08\Analysis'
    videoFilename = "cluster" + str(pick_cluster) + ".avi"
    extra_average_microVolts = np.average(all_cells_ivm_filtered_data[pick_cluster][:,:,:],axis=2) * voltage_step_size* scale_uV
    num_samples = extra_average_microVolts.shape[1]
    time_axis = np.arange(-(num_samples/2)/20000.0,(num_samples/2)/20000,1/20000.0)
    plot_video_topoplot_with_juxta256(extra_average_microVolts, time_axis, channel_positions,
                                      times_to_plot=[-0.002, 0.002],time_window=0.000060,time_step=0.000060,
                                      sampling_freq=20000,
                                      zlimits=[np.min(extra_average_microVolts), np.max(extra_average_microVolts) * 1.1],
                                      filename= os.path.join(rootDir,videoFilename))






#plot impedance w colormap

impedancevalues = np.genfromtxt(r'Z:\j\Neuroseeker256ch\Type1_Probe6\2017-02-08\impedance-2017-02-08.txt')
impedancevalues = np.genfromtxt(r'Z:\j\Neuroseeker256ch_probe6\2017-02-16\impedance-2017-02-16.txt')
impedancevalues = np.genfromtxt(r'Z:\j\Neuroseeker256ch_probe6\2017_02-22\impedance-2017-02-22.txt')


impedancevalues_MOhm= impedancevalues*0.000001
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
channel_positions=polytrode_256channels(bad_channels=[-1])
image, scat = plot_topoplot256(ax1, channel_positions,impedancevalues_MOhm, show=False, interpolation_method='none', gridscale=5, zlimits=[np.min(impedancevalues_MOhm), np.max(impedancevalues_MOhm)])
ax1.set_yticks([], minor=False)
ax1.set_xticks([], minor=False)
plt.colorbar(mappable=image, ticks = np.arange(np.min(impedancevalues_MOhm), np.max(impedancevalues_MOhm), 1))
plt.show()


