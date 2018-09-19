import itertools
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import scipy.signal as signal
import scipy.stats as stats
from matplotlib import mlab
import IO.ephys as ephys


#Plot traces Neuroseeker 256 channels-----------------------------------------------------------------------------------
#File names
#256ch probe in saline 2017-02-08
raw_data_file_ivm = r'Z:\j\Neuroseeker256ch\Type1_Probe6\2017-02-08\Data\Noise saline\amplifier2017-02-07T18_12_22_int16.bin'


#Open data
num_ivm_channels = 256
amp_dtype = np.int16
sampling_freq = 20000
high_pass_freq = 250
filtered_data_type = np.float64
voltage_step_size = 0.195e-6
scale_uV = 1000000
samples = 3000000 #2.5 minutes
raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix [:, 0:samples]
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)


#High-pass filter
def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=100.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)

temp_filtered = highpass(temp_unfiltered, F_HIGH = (sampling_freq/2)*0.95, sampleFreq = sampling_freq, passFreq = high_pass_freq)
temp_filtered_uV = temp_filtered * scale_uV * voltage_step_size


#Plot
offset_microvolt = 100
time_samples = 100000.0 #5 seconds
index1 = np.int(temp_filtered_uV.shape[1]/10*9)
index2 = np.int(index1 + time_samples)
plt.figure()
for i in np.arange(0, np.shape(temp_filtered_uV)[0]):
    plt.plot(temp_filtered_uV[i,index1:index2].T + (offset_microvolt*i))
    plt.show()




#Calculate noise--------------------------------------------------------------------------------------------------------
noise_median = np.median(np.abs(temp_filtered_uV)/0.6745, axis=1)
noise_median_average = np.average(noise_median)
noise_median_stdv = stats.sem(noise_median)
print('#------------------------------------------------------')
print('Noise_Median:'+ str(noise_median))
print('Noise_Median_average:'+ str(noise_median_average))
print('Noise_Median_stdv:'+ str(noise_median_stdv))
print('#------------------------------------------------------')
#analysis_folder =r'Z:\j\Neuroseeker256ch\Type1_Probe6\2017-02-08\Data\Noise saline'
#analysis_folder =r'Z:\j\Neuroseeker256ch\Type1_Probe6\2017-02-08\Datakilosort\Nfilt_Test\nfilt512\amplifier2017-02-08T15_34_04'
#analysis_folder =r'Z:\j\Neuroseeker256ch\Type1_Probe6\2017-02-08\Datakilosort\Nfilt_Test\nfilt512'
#analysis_folder =r'Z:\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\2017-02-22\Datakilosort\amplifier2017-02-23T14_38_33'
analysis_folder =r'Z:\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\2017-02-16\Datakilosort\amplifier2017-02-16T15_37_59'
filename_Median = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_Median' + '.npy')
np.save(filename_Median, noise_median)



#Plot colormaps---------------------------------------------------------------------------------------------------------
def triggerline(x, **kwargs):
    if x is not None:
        ylim = plt.ylim()
        plt.vlines(x,ylim[0],ylim[1],alpha=0.4)

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

    electrode_amplifier_index_on_grid = np.genfromtxt(r"Z:\labs2\kampff\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\2017-02-08\chanmap256.csv", delimiter=",")

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

    channel_positions = channel_positions.sort_values('Numbers', ascending=True)
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

#Figure 3 Supplementary material----------------------------------------------------------------------------------------
#Plot noise w colormap
Medianvalues = np.load(filename_Median)
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
channel_positions = polytrode_256channels(bad_channels=[-1])
image, scat = plot_topoplot256(ax1, channel_positions, Medianvalues, show=False, interpolation_method='none', gridscale=5, zlimits=[np.min(Medianvalues), np.max(Medianvalues)])
ax1.set_yticks([], minor=False)
ax1.set_xticks([], minor=False)
plt.colorbar(mappable=image, ticks = np.arange(np.min(Medianvalues), np.max(Medianvalues), 0.1))
plt.title('Noise_Median_values')
plt.show()
#Plot impedance w colormap
impedancevalues = np.genfromtxt(r"Z:\labs2\kampff\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\2017-02-08\impedance-2017-02-08.txt")
impedancevalues_MOhm = impedancevalues*0.000001
impedancevalues_MOhm[impedancevalues_MOhm > 2] = 0
impedancevalues_MOhm_norm_float = impedancevalues_MOhm / max(impedancevalues_MOhm)
impedancevalues_MOhm_norm = np.round(impedancevalues_MOhm_norm_float, 1)
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
channel_positions=polytrode_256channels(bad_channels=[-1])
image, scat = plot_topoplot256(ax1, channel_positions,impedancevalues_MOhm_norm, show=False, interpolation_method='none', gridscale=5, zlimits=[np.min(impedancevalues_MOhm_norm), np.max(impedancevalues_MOhm_norm)])
ax1.set_yticks([], minor=False)
ax1.set_xticks([], minor=False)
plt.title('Impedance_values_saline')
plt.colorbar(mappable=image, ticks = np.arange(np.min(impedancevalues_MOhm_norm), np.max(impedancevalues_MOhm_norm), 0.1))
plt.show()














































#code not in use

#256ch probe in acute recordings 2017-02-08
#co2
raw_data_file_ivm = r'Z:\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\2017-02-08\Datakilosort\Nfilt_Test\nfilt512\amplifier2017-02-08T15_34_04\amplifier2017-02-08T15_34_04_int16.bin'
#c05
raw_data_file_ivm =r"Z:\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\2017-02-22\Datakilosort\amplifier2017-02-23T14_38_33\amplifier2017-02-23T14_38_33_int16.bin"
#cr1
raw_data_file_ivm =r"Z:\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\2017-02-16\Datakilosort\amplifier2017-02-16T15_37_59\amplifier2017-02-16T15_37_59_int16.bin"


#Protocol 2: RMS noise level for all channels
def RMS_calculation(data):

    NumSites = 256
    RMS = np.zeros(NumSites)

    for i in range(NumSites):
        RMS[i] = np.sqrt((1/len(data[i]))*np.sum(data[i]**2))

    return RMS

noise_rms = RMS_calculation(temp_filtered_uV)
noise_rms_average = np.average(noise_rms)
noise_rms_stdv = stats.sem(noise_rms)

print('#------------------------------------------------------')
print('RMS:'+ str(noise_rms))
print('RMS_average:'+ str(noise_rms_average))
print('RMS_average_stdv:'+ str(noise_rms_stdv))
print('#------------------------------------------------------')

analysis_folder =r'Z:\j\Neuroseeker256ch\Type1_Probe6\2017-02-08\Datakilosort\Nfilt_Test\nfilt512'

filename_RMS = os.path.join(analysis_folder + '\\' + str(high_pass_freq) + 'noise_RMS' + '.npy')

np.save(filename_RMS, noise_rms)


analysis_folder =r'Z:\j\Neuroseeker256ch\Type1_Probe6\2017-02-08\Data\Noise saline'
filename_RMS = os.path.join(analysis_folder + '\\' + 'noise_RMS' + '.npy')
np.save(filename_RMS, noise_rms)

#RMS noise w color
RMSvalues = np.load(filename_RMS)
RMSvalues_norm_float= RMSvalues / max(RMSvalues)
RMSvalues_norm = np.round(RMSvalues_norm_float, 1)
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
channel_positions = polytrode_256channels(bad_channels=[-1])
image, scat = plot_topoplot256(ax1, channel_positions, RMSvalues_norm, show=False, interpolation_method='none', gridscale=5, zlimits=[np.min(RMSvalues_norm), np.max(RMSvalues_norm)])
ax1.set_yticks([], minor=False)
ax1.set_xticks([], minor=False)
plt.colorbar(mappable=image, ticks = np.arange(np.min(RMSvalues_norm), np.max(RMSvalues_norm), 0.1))
plt.title('Noise_RMS_values')
plt.show()


#SPD--------------------------------------------------------------------------------------------------------------------

# All Channels: 256channels
plt.figure()
Pxx_dens_g =[]
for i in np.arange(num_ivm_channels):
    Pxx_dens, f = mlab.psd(temp_filtered_uV[i,:], sampling_freq, sampling_freq)
    Pxx_dens_g.append(Pxx_dens)
    plt. semilogy(f, Pxx_dens, color='r', linewidth=1, alpha=0.2)
    plt.semilogx()
    #plt.ylim([1e-5,1000000])
    #plt.xlim([2,17000])
    #plt.ylim([0.5e-3,100])
    #plt.xlim([1,6000])
    plt.xlabel('frequency(Hz)')
    plt.ylabel('PSD (uV^2/Hz)')

Pxx_dens_g_matrix = np.array(Pxx_dens_g)
Pxx_dens_g_tranp = Pxx_dens_g_matrix.T
np.savetxt(r'E:\Paper Impedance\256chNeuroseeker\Noise\Saline\SPD'+ 'power_' +  str(high_pass_freq) + 'Hz' + '.txt', Pxx_dens_g_tranp, delimiter=',')
np.savetxt(r'E:\Paper Impedance\256chNeuroseeker\Noise\Saline\SPD' + 'frequency_' +  str(high_pass_freq) + 'Hz' +'.txt', f, delimiter=',')

filename_power = os.path.join(r'E:\Paper Impedance\256chNeuroseeker\Noise\Saline\SPD' + '\\' + 'power_' + str(high_pass_freq) + 'Hz' +'.npy')
np.save(filename_power, Pxx_dens_g_tranp)

#Integration of power across frequencies
power_channels = []
for i in np.arange(num_ivm_channels):
    integrate_power = np.trapz(Pxx_dens_g_matrix[i][500:7500])
    power_channels.append(integrate_power)

power_matrix = np.array(power_channels)
power = np.append(np.arange(num_ivm_channels).reshape(num_ivm_channels,1), power_matrix.reshape(num_ivm_channels,1), 1)

np.savetxt(r'E:\Paper Impedance\256chNeuroseeker\Noise\Saline\SPD' + 'Powerintegration_' + str(high_pass_freq) + 'Hz' + '.txt', power, delimiter=',')





#GOD channels: 239channels
bad_channels = [7, 8, 19, 28, 30, 32, 34, 94, 131, 137, 148, 149, 151, 162, 171, 232, 31]
channels = np.delete(np.arange(num_ivm_channels), bad_channels)

plt.figure()
Pxx_dens_g =[]
for i in channels:
    Pxx_dens, f = mlab.psd(temp_filtered_uV[i,:], sampling_freq, sampling_freq)
    Pxx_dens_g.append(Pxx_dens)
    plt. semilogy(f, Pxx_dens, color='r', linewidth=1, alpha=0.2)
    plt.semilogx()
    #plt.ylim([1e-5,1000000])
    #plt.xlim([2,17000])
    #plt.ylim([0.5e-3,100])
    #plt.xlim([1,6000])
    plt.xlabel('frequency(Hz)')
    plt.ylabel('PSD (uV^2/Hz)')

Pxx_dens_g_matrix = np.array(Pxx_dens_g)
Pxx_dens_g_tranp = Pxx_dens_g_matrix.T
np.savetxt(r'E:\Paper Impedance\256chNeuroseeker\Noise\Saline\SPD'+ 'power_' +  str(high_pass_freq) + 'Hz' + '.txt', Pxx_dens_g_tranp, delimiter=',')
np.savetxt(r'E:\Paper Impedance\256chNeuroseeker\Noise\Saline\SPD' + 'frequency_' +  str(high_pass_freq) + 'Hz' +'.txt', f, delimiter=',')

filename_power = os.path.join(r'E:\Paper Impedance\256chNeuroseeker\Noise\Saline\SPD' + '\\' + 'power_' + str(high_pass_freq) + 'Hz' +'.npy')
np.save(filename_power, Pxx_dens_g_tranp)

#Integration of power across frequencies
power_channels = []
for i in np.arange(channels.shape[0]):
    integrate_power = np.trapz(Pxx_dens_g_matrix[i][500:7500])
    power_channels.append(integrate_power)

power_matrix = np.array(power_channels)
power = np.append(channels.reshape(239,1), power_matrix.reshape(239,1), 1)

np.savetxt(r'E:\Paper Impedance\256chNeuroseeker\Noise\Saline\SPD' + 'Powerintegration_' + str(high_pass_freq) + 'Hz' + '.txt', power, delimiter=',')



#Noise in saline w colormap
filename_RMS_saline = r'E:\Paper Impedance\256chNeuroseeker\Noise\Saline\Noise saline\250noise_RMS.npy'
RMSvalues_saline = np.load(filename_RMS_saline)
RMSvalues_saline =noise_median
RMSvalues_norm_float= RMSvalues_saline / max(RMSvalues_saline)
RMSvalues_norm = np.round(RMSvalues_norm_float, 1)
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
channel_positions = polytrode_256channels(bad_channels=[-1])
image, scat = plot_topoplot256(ax1, channel_positions, RMSvalues_norm, show=False, interpolation_method='none', gridscale=5, zlimits=[np.min(RMSvalues_norm), np.max(RMSvalues_norm)])
ax1.set_yticks([], minor=False)
ax1.set_xticks([], minor=False)
plt.colorbar(mappable=image, ticks = np.arange(np.min(RMSvalues_norm), np.max(RMSvalues_norm), 0.1))
plt.title('Noise_RMS_values_saline')
plt.show()






#Scaterplot impedance vs noise saline and noise brain-------------------------------------------------------------------
plt.figure()
plt.scatter(np.arange(num_ivm_channels), RMSvalues_saline, color='g')
plt.scatter(np.arange(num_ivm_channels), RMSvalues_brain, color='r')
plt.scatter(np.arange(num_ivm_channels),impedancevalues_MOhm)
triggerline(np.arange(num_ivm_channels))



#Plot noise in brain w colormap
#filename_RMS_brain = r'E:\Paper Impedance\256chNeuroseeker\Noise\amplifier2017-02-08T15_34_04\250noise_RMS.npy'
filename_RMS_brain =r"Z:\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\2017-02-22\Datakilosort\amplifier2017-02-23T14_38_33\250noise_Median.npy"
RMSvalues_brain = np.load(filename_RMS_brain)
#RMSvalues_norm_float= RMSvalues_brain / max(RMSvalues_brain)
#RMSvalues_norm = np.round(RMSvalues_norm_float, 1)
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
channel_positions = polytrode_256channels(bad_channels=[-1])
image, scat = plot_topoplot256(ax1, channel_positions, RMSvalues_brain, show=False, interpolation_method='none', gridscale=5, zlimits=[np.min(RMSvalues_brain), np.max(RMSvalues_brain)])
ax1.set_yticks([], minor=False)
ax1.set_xticks([], minor=False)
plt.colorbar(mappable=image)
plt.title('Noise_RMS_values_15_34_04')
plt.show()

#Plot noise in brain w colormap
#filename_RMS_brain = r'E:\Paper Impedance\256chNeuroseeker\Noise\amplifier2017-02-08T15_34_04\250noise_RMS.npy'
filename_RMS_brain =r"Z:\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\2017-02-22\Datakilosort\amplifier2017-02-23T14_38_33\250noise_Median.npy"
RMSvalues_brain = np.load(filename_RMS_brain)
#RMSvalues_norm_float= RMSvalues_brain / max(RMSvalues_brain)
#RMSvalues_norm = np.round(RMSvalues_norm_float, 1)
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
channel_positions = polytrode_256channels(bad_channels=[-1])
image, scat = plot_topoplot256(ax1, channel_positions, RMSvalues_brain, show=False, interpolation_method='none', gridscale=5, zlimits=[np.min(RMSvalues_brain), np.max(RMSvalues_brain)])
ax1.set_yticks([], minor=False)
ax1.set_xticks([], minor=False)
plt.colorbar(mappable=image)
plt.title('Noise_RMS_values_15_34_04')
plt.show()
