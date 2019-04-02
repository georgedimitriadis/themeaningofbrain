import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

import Layouts.Probes.klustakwik_prb_generator as prb_gen


# Return all_electrodes

def create_128channels_imec_prb(filename=None, bad_channels=None):

    r1 = np.array([103,	101, 99, 97,	95,	93,	91,	89,	87,	70,	66,	84,	82,	80,	108,	110,	47,	45,	43,	41,	1,61,	57,
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
        prb_gen.generate_prb_file(filename=filename, all_electrodes_array=all_electrodes, channel_number=128)

    return all_electrodes


# Colormap from Pylyb

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


# Plot a line at time x

def triggerline(x):
    if x is not None:
        ylim = plt.ylim()
        plt.vlines(x,ylim[0],ylim[1])


# Plot all JUXTA spikes w average

scale_mV = 1000
scale_ms = 1000

def plot_alltriggers_average_juxta(all_cells_patch_data, low_pass_freq=5000, y_offset_mV=0.5):
        for i in np.arange(0, len(good_cells)):
            num_samples=all_cells_patch_data[good_cells[i]].shape[0]
            sample_axis= np.arange(-((num_samples/2)),((num_samples/2)))
            time_axis= sample_axis/30000
            plt.figure(); plt.plot(time_axis*scale_ms, all_cells_patch_data[good_cells[i]]*scale_mV, alpha=0.1, color='0.3')
            patch_average= np.average(all_cells_patch_data[good_cells[i]],axis=1)
            plt.plot(time_axis*scale_ms, patch_average*scale_mV, linewidth='3', color='g')
            plt.xlim(-2, 2)
            plt.ylim(np.min(patch_average*scale_mV)-y_offset_mV,np.max(patch_average*scale_mV)+y_offset_mV )
            plt.ylabel('Voltage (mV)', fontsize=20)
            plt.xlabel('Time (ms)', fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            triggerline(0)

# Plot 32channel averages overlaid

voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000
#voltage_step_size = 1
#scale_uV = 1

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
        extra_average_microVolts = extra_average_V * scale_uV * voltage_step_size
        plt.figure()
        for m in np.arange(np.shape(extra_average_microVolts)[0]):
            colorVal=scalarMap.to_rgba(np.shape(extra_average_microVolts)[0]-m)
            plt.plot(time_axis*scale_ms,extra_average_microVolts[sites_order_geometry[m],:].T, color=colorVal)
            plt.xlim(-2, 2) #window 4ms
            plt.ylim(np.min(extra_average_microVolts)-yoffset,np.max(extra_average_microVolts)+yoffset )
            plt.ylabel('Voltage (\u00B5V)', fontsize=20)
            plt.xlabel('Time (ms)',fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            triggerline(0)

# Plot 32channels averages in space

voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000
#voltage_step_size = 1
#scale_uV = 1

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
        extra_average_microVolts= np.average(all_cells_ivm_filtered_data[good_cells[m]][:,:,:],axis=2) * voltage_step_size* scale_uV
        num_samples=extra_average_microVolts.shape[1]
        sample_axis= np.arange(-(num_samples/2),(num_samples/2))
        time_axis= sample_axis/30000
        for i in np.arange(32):
            plt.subplot(22,3,subplot_order [i])
            colorVal=scalarMap.to_rgba(31-i)
            plt.plot(time_axis*scale_ms, extra_average_microVolts[sites_order_geometry[i],:],color=colorVal)
            plt.ylim(np.min(np.min(extra_average_microVolts, axis=1))- yoffset, np.max(np.max(extra_average_microVolts, axis=1))+ yoffset)
            plt.xlim(-2, 2)
            plt.axis("OFF")


# Plot 128channels averages overlaid
all_electrodes = create_128channels_imec_prb()
voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000

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
        extra_average_microVolts = extra_average_V* scale_uV
        plt.figure()
        for m in np.arange(np.shape(extra_average_microVolts)[0]):
            colorVal=scalarMap.to_rgba(np.shape(extra_average_microVolts)[0]-m)
            plt.plot(time_axis*scale_ms,extra_average_microVolts[subplot_number_array[:,m],:].T, color=colorVal)
            plt.xlim(-2, 2) #window 4ms
            plt.ylim(np.min(extra_average_microVolts)-yoffset,np.max(extra_average_microVolts)+yoffset )
            plt.ylabel('Voltage (\u00B5V)', fontsize=20)
            plt.xlabel('Time (ms)',fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            triggerline(0)

# Plot 128channels averages in space
voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000

def plot_average_extra_geometry(all_cells_ivm_filtered_data, electrode_structure=all_electrodes):

    origin_cmap= plt.get_cmap('hsv')
    shrunk_cmap = shiftedColorMap(origin_cmap, start=0.6, midpoint=0.7, stop=1, name='shrunk')
    cm=shrunk_cmap
    cNorm=colors.Normalize(vmin=0,vmax=128)
    scalarMap= cmx.ScalarMappable(norm=cNorm,cmap=cm)
    for m in np.arange(0, len(good_cells)):
        plt.figure(m+1)
        subplot_number_array = electrode_structure.reshape(1,128)
        extra_average_microVolts= np.average(all_cells_ivm_filtered_data[good_cells[m]][:,:,:],axis=2)* scale_uV * voltage_step_size
        num_samples=extra_average_microVolts.shape[1]
        sample_axis= np.arange(-(num_samples/2),(num_samples/2))
        time_axis= sample_axis/30000
        for i in np.arange(1, subplot_number_array.shape[1]+1):
            plt.subplot(32,4,i)
            colorVal=scalarMap.to_rgba(subplot_number_array.shape[1]-i)
            plt.plot(time_axis*scale_ms, extra_average_microVolts[np.int(subplot_number_array[:,i-1]),:],color=colorVal)
            plt.ylim(np.min(np.min(extra_average_microVolts, axis=1)), np.max(np.max(extra_average_microVolts, axis=1)))
            plt.xlim(-2, 2)
            plt.axis("OFF")



