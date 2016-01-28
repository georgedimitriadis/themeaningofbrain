



import os
import IO.ephys as ephys
import mne.filter as filters
import matplotlib.pyplot as plt
import random
from matplotlib import colors
import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import warnings
import numpy as np
import Layouts.Probes.klustakwik_prb_generator as prb_gen
import pandas as pd
import itertools
import numpy as np
import matplotlib as mpl
from scipy import stats
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
import scipy.signal as signal
import matplotlib.colors as mcolors
import math
import scipy.stats as stats

from ExperimentSpecificCode.Paired_128ch_JPak_96_to_99.Joana_Neto.Tsn32 import patch_data
from ShuttlingAnalysis.paper.activityplots import clearcollection

# return all_electrodes and channel_positions
def create_128channels_imec_prb(filename=None, bad_channels=None):

    r1 = np.array([103,	101, 99,	97,	95,	93,	91,	89,	87,	70,	66,	84,	82,	80,	108,	110,	47,	45,	43,	41,	1,61,	57,
                   36,	34,	32,	30,	28,	26,	24,	22,	20])
    r2 = np.array([106, 104, 115, 117, 119, 121, 123, 125, 127, 71, 67, 74, 76, 78, 114, 112,
                   49, 51, 53, 55, 2, 62, 58, 4, 6, 8, 10, 12, 14, 21, 19, 16])
    r3 = np.array([102, 100, 98, 96, 94, 92, 90, 88, 86, 72, 68, 65, 85, 83, 81, 111, 46, 44, 42, 40, 38, 63, 59,
                   39, 37, 35, 33, 31, 29, 27, 25, 23])
    r4 = np.array([109, 107, 105, 116, 118, 120, 122, 124, 126, 73, 69, 64, 75, 77, 79, 113,
                   48, 50, 52, 54, 56, 0, 60, 3, 5, 7, 9, 11, 13, 15, 18,-1])

    all_electrodes_concat = np.concatenate((r1, r2, r3, r4))
    all_electrodes = all_electrodes_concat.reshape((4, 32))
    all_electrodes = np.flipud(all_electrodes.T)


    if filename is not None:
        prb_gen.generate_prb_file(filename=filename, all_electrodes_array=all_electrodes)


    electrode_coordinate_grid = list(itertools.product(np.arange(1, 5), np.arange(1, 33)))

    electrode_coordinate_grid = list(itertools.product(np.arange(1, 33), np.arange(1, 5)))

    electrode_coordinate_grid = [tuple(reversed(x)) for x in electrode_coordinate_grid]
    electrode_amplifier_index_on_grid = all_electrodes_concat
    electrode_amplifier_name_on_grid = np.array(["Int" + str(x) for x in electrode_amplifier_index_on_grid])
    indices_arrays = [electrode_amplifier_index_on_grid.tolist(), electrode_amplifier_name_on_grid.tolist()]
    indices_tuples = list(zip(*indices_arrays))
    channel_position_indices = pd.MultiIndex.from_tuples(indices_tuples, names=['Numbers', 'Strings'])
    channel_positions = pd.Series(electrode_coordinate_grid, index=channel_position_indices)
    channel_positions.columns = ['Positions']
    channel_positions = channel_positions.reset_index(level=None, drop=False, name='Positions')
    if bad_channels is not None:
        channel_positions = channel_positions[~channel_positions.Numbers.isin(bad_channels)]

    return all_electrodes, channel_positions



#change colormap from Pylyb
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


# plot a line

def triggerline(x):
    if x is not None:
        ylim = plt.ylim()
        plt.vlines(x,ylim[0],ylim[1])


# plot JUXTA spikes w average
  def plot_alltriggers_average_juxta(all_cells_patch_data, low_pass_freq=5000, y_offset_mV=0.5):
        for i in np.arange(0, len(good_cells)):
            num_samples=all_cells_patch_data[good_cells[i]].shape[0]
            time_sec= num_samples/30000
            sample_axis= np.arange(-((num_samples/2)),((num_samples/2)))
            time_axis= sample_axis/30000
            #plt.figure(); plt.plot(time_axis*1000, all_cells_patch_data[good_cells[i]]*100, alpha=0.1, color='0.3')
            plt.figure(); plt.plot(time_axis*1000, all_cells_patch_data[good_cells[i]]*1000, alpha=0.1, color='0.3')
            patch_average= np.average(all_cells_patch_data[good_cells[i]],axis=1)
            patch_average =filters.low_pass_filter(patch_average, sampling_freq,low_pass_freq,method='iir',iir_params=iir_params)
            #plt.plot(time_axis*1000, patch_average*100, linewidth='3', color='g')
            plt.plot(time_axis*1000, patch_average*1000, linewidth='3', color='g')
            plt.xlim(-2, 2)
            #plt.ylim(np.min(patch_average*100)-y_offset_mV,np.max(patch_average*100)+y_offset_mV )
            plt.ylim(np.min(patch_average*1000)-y_offset_mV,np.max(patch_average*1000)+y_offset_mV )
            plt.ylabel('Voltage (mV)', fontsize=20)
            plt.xlabel('Time (ms)', fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            triggerline(0)



# electrode_structure is length x width #of electrodes with the number of the intan data in each element
def spread_data(data, electrode_structure, col_spacing=3, row_spacing=0.5):

    data_baseline = (data.T - np.median(data, axis=1).T).T
    new_data = np.zeros(shape=(np.shape(data)[0], np.shape(data)[1]))

    num_of_rows = np.shape(electrode_structure)[0]
    num_of_cols = np.shape(electrode_structure)[1]


    stdv = np.average(np.std(data_baseline, axis=1), axis=0)

    col_spacing = col_spacing * stdv
    row_spacing = row_spacing * stdv
    for r in np.arange(0, num_of_rows):
        for c in np.arange(0, num_of_cols):
            new_data[electrode_structure[r, c], :] = data_baseline[electrode_structure[r, c], :] - \
                                                     r * col_spacing - \
                                                     c * row_spacing

    return new_data

#32 channel average
voltage_step_size = 0.195e-6

def plot_average_extra(all_cells_ivm_filtered_data, yoffset=1):

    origin_cmap= plt.get_cmap('hsv')
    shrunk_cmap = shiftedColorMap(origin_cmap, start=0.6, midpoint=0.7, stop=1, name='shrunk')
    cm=shrunk_cmap
    cNorm=colors.Normalize(vmin=0,vmax=32)
    scalarMap= cmx.ScalarMappable(norm=cNorm,cmap=cm)
    sites_order_geometry= [0,31,24,7,1,21,10,30,25,6,15,20,11,16,26,5,14,19,12,17,27,4,8,18,13,23,28,3,9,29,2,22]
    for i in np.arange(0, len(good_cells)):
        extra_average_V = np.average(all_cells_ivm_filtered_data[good_cells[i]][:,:,:],axis=2)
        num_samples=extra_average_V.shape[1]
        sample_axis= np.arange(-(num_samples/2),(num_samples/2))
        time_axis= sample_axis/30000
        #extra_average_microVolts = extra_average_V
        extra_average_microVolts = extra_average_V* 1000000*voltage_step_size
        plt.figure()
        for m in np.arange(np.shape(extra_average_microVolts)[0]):
            colorVal=scalarMap.to_rgba(np.shape(extra_average_microVolts)[0]-m)
            plt.plot(time_axis*1000,extra_average_microVolts[sites_order_geometry[m],:].T, color=colorVal)
            plt.xlim(-2, 2)
            plt.ylim(np.min(extra_average_microVolts)-yoffset,np.max(extra_average_microVolts)+yoffset )
            plt.ylabel('Voltage (\u00B5V)', fontsize=20)
            plt.xlabel('Time (ms)',fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            triggerline(0)

#128 channles average


all_electrodes, channel_positions=create_128channels_imec_prb()

voltage_step_size = 0.195e-6
def plot_average_extra(all_cells_ivm_filtered_data, electrode_structure=all_electrodes, yoffset=1):

    origin_cmap= plt.get_cmap('hsv')
    shrunk_cmap = shiftedColorMap(origin_cmap, start=0.6, midpoint=0.7, stop=1, name='shrunk')
    cm=shrunk_cmap
    cNorm=colors.Normalize(vmin=0,vmax=128)
    scalarMap= cmx.ScalarMappable(norm=cNorm,cmap=cm)

    subplot_number_array = electrode_structure.reshape(1,128)
    for i in np.arange(0, len(good_cells)):
        extra_average_V = np.average(all_cells_ivm_filtered_data[good_cells[i]][:,:,:],axis=2)* voltage_step_size
        num_samples=extra_average_V.shape[1]
        sample_axis= np.arange(-(num_samples/2),(num_samples/2))
        time_axis= sample_axis/30000
        extra_average_microVolts = extra_average_V* 1000000
        plt.figure()
        for m in np.arange(np.shape(extra_average_microVolts)[0]):
            colorVal=scalarMap.to_rgba(np.shape(extra_average_microVolts)[0]-m)
            plt.plot(time_axis*1000,extra_average_microVolts[subplot_number_array[:,m],:].T, color=colorVal)
            plt.xlim(-2, 2)
            plt.ylim(np.min(extra_average_microVolts)-yoffset,np.max(extra_average_microVolts)+yoffset )
            plt.ylabel('Voltage (\u00B5V)', fontsize=20)
            plt.xlabel('Time (ms)',fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            triggerline(0)

#####128ch in space

voltage_step_size = 0.195e-6

def plot_average_extra_geometry(all_cells_ivm_filtered_data, electrode_structure=all_electrodes):

    origin_cmap= plt.get_cmap('hsv')
    shrunk_cmap = shiftedColorMap(origin_cmap, start=0.6, midpoint=0.7, stop=1, name='shrunk')
    cm=shrunk_cmap
    cNorm=colors.Normalize(vmin=0,vmax=128)
    scalarMap= cmx.ScalarMappable(norm=cNorm,cmap=cm)
    for m in np.arange(0, len(good_cells)):
        plt.figure(m+1)
        subplot_number_array = electrode_structure.reshape(1,128)
        extra_average_microVolts= np.average(all_cells_ivm_filtered_data[good_cells[m]][:,:,:],axis=2)* 1000000 * voltage_step_size
        num_samples=extra_average_microVolts.shape[1]
        sample_axis= np.arange(-(num_samples/2),(num_samples/2))
        time_axis= sample_axis/30000
        for i in np.arange(1, subplot_number_array.shape[1]+1):
            plt.subplot(32,4,i)
            colorVal=scalarMap.to_rgba(subplot_number_array.shape[1]-i)
            plt.plot(time_axis*1000, extra_average_microVolts[np.int(subplot_number_array[:,i-1]),:],color=colorVal)
            plt.ylim(np.min(np.min(extra_average_microVolts, axis=1)), np.max(np.max(extra_average_microVolts, axis=1)))
            plt.xlim(-2, 2)
            plt.axis("OFF")



voltage_step_size = 0.195e-6

#32channels in space

def plot_average_extra_geometry(all_cells_ivm_filtered_data, yoffset=1):


    sites_order_geometry= [0,31,24,7,1,21,10,30,25,6,15,20,11,16,26,5,14,19,12,17,27,4,8,18,13,23,28,3,9,29,2,22]
    subplot_order = [2,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65]
    origin_cmap= plt.get_cmap('hsv')
    shrunk_cmap = shiftedColorMap(origin_cmap, start=0.6, midpoint=0.7, stop=1, name='shrunk')
    cm=shrunk_cmap
    cNorm=colors.Normalize(vmin=-4,vmax=28)
    scalarMap= cmx.ScalarMappable(norm=cNorm,cmap=cm)
    for m in np.arange(0, len(good_cells)):
        plt.figure(m+1)
        #extra_average_microVolts= np.average(all_cells_ivm_filtered_data[good_cells[m]][:,:,:],axis=2)
        extra_average_microVolts= np.average(all_cells_ivm_filtered_data[good_cells[m]][:,:,:],axis=2) * voltage_step_size* 1000000
        num_samples=extra_average_microVolts.shape[1]
        sample_axis= np.arange(-(num_samples/2),(num_samples/2))
        time_axis= sample_axis/30000
        for i in np.arange(32):
            plt.subplot(22,3,subplot_order [i])
            colorVal=scalarMap.to_rgba(31-i)
            plt.plot(time_axis*1000, extra_average_microVolts[sites_order_geometry[i],:],color=colorVal)
            plt.ylim(np.min(np.min(extra_average_microVolts, axis=1))- yoffset, np.max(np.max(extra_average_microVolts, axis=1))+ yoffset)
            plt.xlim(-2, 2)
            plt.axis("OFF")


#P2P, MIN ans MAX

def peaktopeak(all_cells_ivm_filtered_data, windowSize=60):

    for i in np.arange(0, len(good_cells)):
        extra_average_V = np.average(all_cells_ivm_filtered_data[good_cells[i]][:,:,:],axis=2)
        NumSamples=extra_average_V.shape[1]
        #extra_average_microVolts = extra_average_V
        extra_average_microVolts = extra_average_V * 1000000 * voltage_step_size
        NumSites=np.size(extra_average_microVolts,axis = 0)
        lowerBound=int(NumSamples/2.0-windowSize/2.0)
        upperBound=int(NumSamples/2.0+windowSize/2.0)

        #shift=(upperBound-lowerBound)
        #if shift%2 != 0:
        #    shift += 1
        #shift /= 2

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
        stdv = stats.sem(all_cells_ivm_filtered_data[good_cells[i]][:,:,:], axis=2)
        stdv = stdv * voltage_step_size * 1000000

        for b in range(NumSites):
           stdv_minimos[b]= stdv[b, argminima[b]]
           stdv_maximos[b]= stdv[b, argmaxima[b]]

        error =  np.sqrt((stdv_minimos * stdv_minimos)+ (stdv_maximos*stdv_maximos))

        #negative_error= (p2p)-(error)
        #positive_error= (p2p)+(error)


        np.save(os.path.join(analysis_folder,'stdv_minimos_EXTRA_Cell'+ good_cells[i] + '.npy'), stdv_minimos)
        np.save(os.path.join(analysis_folder,'stdv_maximos_EXTRA_Cell'+ good_cells[i] + '.npy'), stdv_maximos)
        np.save(os.path.join(analysis_folder,'stdv_average_EXTRA_Cell'+ good_cells[i] + '.npy'), stdv)
        np.save(os.path.join(analysis_folder,'error_EXTRA_Cell'+ good_cells[i] + '.npy'), error)
        np.save(os.path.join(analysis_folder,'p2p_EXTRA_Cell'+ good_cells[i] + '.npy'), p2p)
        np.save(os.path.join(analysis_folder,'minima_EXTRA_Cell'+ good_cells[i] + '.npy'), minima)
        np.save(os.path.join(analysis_folder,'maxima_EXTRA_Cell'+ good_cells[i] + '.npy'), maxima)
        np.save(os.path.join(analysis_folder,'argmaxima_EXTRA_Cell'+ good_cells[i] + '.npy'), argmaxima)
        np.save(os.path.join(analysis_folder,'argminima_EXTRA_Cell'+ good_cells[i] + '.npy'), argminima)

    #return argmaxima, argminima, maxima, minima, p2p

#AMP, ERROR NO ORDER

filename = r'D:\Protocols\PairedRecordings\Neuroseeker128\Data\2015-08-28\Analysis'
amplitudes = np.load(os.path.join(filename + '\p2p_EXTRA_cell9.npy'))
error = np.load(os.path.join(filename +  '\error_EXTRA_cell9.npy'))

min_ampl= amplitudes.min()
max_ampl= amplitudes.max()
channel = amplitudes.argmax()
print(min_ampl)
print(max_ampl)
print(channel)
closest_channel = 22

print(amplitudes[closest_channel])
print(error[closest_channel])

####heatmapp with amplitude w ORDER
def heatmapp_amplituide(all_cells_ivm_filtered_data, good_cells_number = 0, windowSize=60):

    voltage_step_size = 0.195e-6
    extra_average_V = np.average(all_cells_ivm_filtered_data[good_cells[good_cells_number]][:,:,:],axis=2) * voltage_step_size
    extra_average_microVolts = extra_average_V * 1000000
    orderedSites = all_electrodes.reshape(1,128)
    NumSamples=extra_average_V.shape[1]
    NumSites=np.size(extra_average_microVolts,axis = 0)
    lowerBound=int(NumSamples/2.0-windowSize/2.0)
    upperBound=int(NumSamples/2.0+windowSize/2.0)
    amplitude = np.zeros(128)
    for j in np.arange(128):
        amplitude[j] =  np.max(extra_average_microVolts[orderedSites[0][j],lowerBound:upperBound])-np.min(extra_average_microVolts[orderedSites[0][j],lowerBound:upperBound])
    return amplitude

##EACH CELL AT TIME i=number
i = 2
amplitude = heatmapp_amplituide(all_cells_ivm_filtered_data, good_cells_number = i)

np.save(os.path.join(analysis_folder,'amplitude_EXTRA_Cell'+ good_cells[i] + '.npy'), amplitude)

####scheme of heatmap

B = np.copy(amplitude)

orderedSites = all_electrodes.reshape(1,128)
Bmod = np.reshape(B,(32,4))
fig, ax = plt.subplots()
#ax.set_title('Distances Heatmap',fontsize=10, position=(0.8,1.02))
plt.axis('off')
im = ax.imshow(Bmod, cmap=plt.get_cmap('jet'),vmin = np.min(B),vmax= np.max(B))
cb = fig.colorbar(im,ticks = [np.min(B), 0,np.max(B)])
cb.ax.tick_params(labelsize = 20)
plt.show()
































def plot_average_extra_offset(all_cells_ivm_filtered_data, electrode_structure=all_electrodes, yoffset=10):
    for m in np.arange(0, len(good_cells)):
        extra_average_microVolts= np.average(all_cells_ivm_filtered_data[good_cells[m]][:,:,:],axis=2)* 1000000 * voltage_step_size
        new_data= spread_data(extra_average_microVolts,electrode_structure=all_electrodes,col_spacing=40,row_spacing=5 )
        num_samples=extra_average_microVolts.shape[1]
        sample_axis= np.arange(-(num_samples/2),(num_samples/2))
        time_axis= sample_axis/30000
        plt.figure(); plt.plot(time_axis,new_data.T,linewidth=1)
        #plt.ylim(np.min(np.min(extra_average_microVolts, axis=1) - yoffset), np.max(np.max(extra_average_microVolts, axis=1)) + yoffset)
        plt.xlim(-0.001, 0.001)
        triggerline(0)




####old code

#32channels code
def COLOR_plot_trigger_average_mapp_visualization(triggerdata,yaxis=15,xneg=-4, xpos=4, trigger=None,**kwargs):
    global data, x_adc, number_samples, time_sec, resolucao, x_adc

    sites_order_geometry= [0,31,24,7,1,21,10,30,25,6,15,20,11,16,26,5,14,19,12,17,27,4,8,18,13,23,28,3,9,29,2,22]
    subplot_order = [2,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65]
    origin_cmap= plt.get_cmap('hsv')
    shrunk_cmap = shiftedColorMap(origin_cmap, start=0.6, midpoint=0.7, stop=1, name='shrunk')
    cm=shrunk_cmap
    cNorm=colors.Normalize(vmin=-4,vmax=28)
    scalarMap= cmx.ScalarMappable(norm=cNorm,cmap=cm)
    #print scalarMap.get_clim()
    average= np.mean(triggerdata,axis=2)
    number_samples=np.shape(triggerdata)[1]
    time_msec=number_samples/30.0
    x_adc=np.linspace(-(time_msec/2), (time_msec/2), num=number_samples)
    for i in np.arange(32):
       plt.subplot(22,3,subplot_order [i])
       colorVal=scalarMap.to_rgba(31-i)
       plt.plot(x_adc, average[sites_order_geometry[i],:].T, color=colorVal)
       plt.xlim(xneg,xpos)
       plt.ylim(-yaxis,yaxis)

def COLOR_lot_trigger_average_mapp(triggerdata,channeloffset=0,trigger=None,**kwargs):
    global data, x_adc

    origin_cmap= plt.get_cmap('hsv')
    shrunk_cmap = shiftedColorMap(origin_cmap, start=0.6, midpoint=0.7, stop=1, name='shrunk')
    cm=shrunk_cmap
    cNorm=colors.Normalize(vmin=-4,vmax=28)
    scalarMap= cmx.ScalarMappable(norm=cNorm,cmap=cm)
    print scalarMap.get_clim()
    average= np.mean(triggerdata,axis=2)
    number_samples=np.shape(triggerdata)[1]
    time_msec=number_samples/30.0
    x_adc=np.linspace(-(time_msec/2), (time_msec/2), num=number_samples)
    sites_order_geometry= [0,31,24,7,1,21,10,30,25,6,15,20,11,16,26,5,14,19,12,17,27,4,8,18,13,23,28,3,9,29,2,22]
    for i in np.arange(32):
        colorVal=scalarMap.to_rgba(31-i)
        plt.plot(x_adc, average[sites_order_geometry[i],:].T +channeloffset*i, color=colorVal)

