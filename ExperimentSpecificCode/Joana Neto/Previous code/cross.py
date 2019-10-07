import itertools
import os

import matplotlib.cm as cmx
import matplotlib.colors as colors
import pandas as pd
import scipy.signal as signal
import seaborn as sns

from BrainDataAnalysis._Old_Structures import Constants as ct
# import Utilities as ut
import numpy as np
import matplotlib.pyplot as plt

"""
# bad channels:
bad_channels = [12, 13, 18, 19, 27, 34, 37, 43, 47, 51, 53, 57, 61, 63, 66, 68, 69, 73, 75, 77, 89, 90, 91, 92, 103,
                105, 108, 114, 116, 117, 125]

num_channels = 128
amp_dtype = np.uint16

#data_folder = '/home/jesse/Data/recording/2017_03_29-14_57/'
data_folder = '/home/jesse/Data/recording/2017_03_23_recording_probe_272'
fn = 'amplifier2017-03-23T22_02_33.bin'

raw_data_file = os.path.join(data_folder, fn)
raw_data = ephys.load_raw_data(raw_data_file, numchannels=num_channels, dtype=amp_dtype)

syncpath = os.path.join(data_folder,'sync.bin')
counterpath = os.path.join(data_folder,'counter.csv')

"""



ivm_num_channels = 256
#data_folder = r'Z:\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\2017-02-16\Data'
#data_folder = r'Z:\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\2017-02-08\Data'
data_folder = r'Z:\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\2017-02-22\Data'

#data_file = 'amplifier2017-02-16T15_37_59.bin'
#data_file = 'amplifier2017-02-08T21_38_55.bin'
#data_file = 'amplifier2017-02-23T14_38_33.bin'
data_file = 'amplifier2017-02-23T17_29_48.bin'





sampling_freq = 20000
high_pass_freq = 250

def load_raw_data():
    raw_data = np.memmap(os.path.join(data_folder, data_file), dtype=np.uint16)
    raw_data = raw_data.reshape((-1, ivm_num_channels))
    return raw_data.T



def plot_corr_movie(data,ffmpeg_path=None):
    """

    :param data:
    :return:
    """
    import matplotlib.animation as animation

    fs = 20000
    time_bin_size = 1 * fs
    chunks = [ data[:,i:i + time_bin_size] for i in range(0, data.shape[1], time_bin_size)]

    corrmat = np.zeros((255,255,len(chunks)))
    print("Computing correlations... \n")
    print("Working on frame: \n " )
    for idx,chunk in enumerate(chunks):
        print("\t"+ str(idx) + " of " + str(len(chunks)))
        corrmat[:,:,idx] = get_corrmat(chunk)

    def init():
        sns.heatmap(np.zeros((255,255)),square=True,xticklabels=[],yticklabels=[])

    def animate(i,corrmats):
        plt.clf()
        corrmat = corrmats[:,:,i]
        sns.heatmap(corrmat,vmin =-.8, vmax=.8, square=True,xticklabels=[],yticklabels=[])

    fig = plt.figure()
    print("Constructing animation... ")
    anim = animation.FuncAnimation(fig,
                                   init_func=init,
                                   func = animate,
                                   fargs=[corrmat],
                                   frames=len(chunks),
                                   repeat = False,
                                   blit=False)

    if ffmpeg_path is not None:
        plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
    else:
        plt.rcParams['animation.ffmpeg_path'] = "C:\\Users\\KAMPFF-LAB_ANALYSIS4\\Downloads\\ffmpeg-20160116-git-d7c75a5-win64-static\\ffmpeg-20160116-git-d7c75a5-win64-static\\bin\\ffmpeg.exe"
    FFwriter = animation.FFMpegWriter()
    filename = "E:\Paper Impedance\Figures\corr_movie.avi"
    anim.save(filename,writer=FFwriter)
    return anim

def get_channel_positions(bad_channels=[]):
    '''
     This function produces a grid with the electrodes positions for the 256 channel probe

     Inputs:
     bad_channels is a list or numpy array with channels you may want to
     disregard.

     Outputs:
     channel_positions is a Pandas Series with the electrodes positions (in
     two dimensions)
     '''

    electrode_coordinate_grid = list(itertools.product(np.arange(0, 17),
                                                       np.arange(0, 15)))

    electrode_amplifier_index_on_grid = np.genfromtxt(r'Z:\j\Joana Neto\Neuroseeker256ch\Type1_Probe6\2017-02-08\chanmap_v2.csv', delimiter=",")

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



def read_sync(syncpath, counterpath):
    sync = np.fromfile(syncpath, dtype=np.uint8)
    syncidx = np.flatnonzero(np.diff(np.int8(sync > 0)) < 0)
    counter = pd.read_csv(counterpath, names=['counter'])
    drops = counter.diff() - 1
    if len(syncidx) != (len(counter) + drops.sum()[0]):
        print
        "WARNING: Number of frames does not match number of sync pulses!"
    matchedpulses = drops.counter.fillna(0).cumsum() + np.arange(len(drops))
    syncidx = syncidx[matchedpulses.astype(np.int32).values]
    syncidx = pd.DataFrame(syncidx, columns=['syncidx'])

    frameidx = np.zeros(sync.shape, dtype=np.int)
    frameidx[0] = -1
    frameidx[syncidx.values] = 1
    frameidx = pd.DataFrame(frameidx, columns=['frame']).cumsum()
    return syncidx, frameidx





def get_average_corrmat_whole_data(data, show_heatmap=True):
    data_shape = data.shape
    chunk_size = 200000
    n_chunks = int(data_shape[1] / chunk_size)

    channel_positions = get_channel_positions()
    num_chans = channel_positions.shape[0]
    #num_chans =15
    corrmat = np.empty((num_chans, num_chans, n_chunks))

    for i in np.arange(0, n_chunks):
        print('Working on chunk ' + str(i + 1) + ' out of ' + str(n_chunks))
        chunk = data[:, i * chunk_size:(chunk_size + i * chunk_size)]
        corrmat[:, :, i] = get_corrmat(chunk, show_heatmap=False)


    corrmat_average = np.mean(corrmat, axis=2)
    corrmat_desvio = np.std(corrmat, axis=2)
    np.save(os.path.join(data_folder,data_file+ '_' + 'cross'  + '.npy'), corrmat)
    np.save(os.path.join(data_folder,data_file+ '_' + 'cross_average'  + '.npy'), corrmat_average)
    np.save(os.path.join(data_folder,data_file+ '_' + 'cross_stdv'  + '.npy'), corrmat_desvio)

    if show_heatmap:
        plot_corr_heatmap(corrmat_average)

    return corrmat



data = load_raw_data()
chunk_size = 20000

data_shape = data.shape
chunk_size = 20000
n_chunks = int(data_shape[1] / chunk_size)

channel_positions = get_channel_positions()
num_chans = channel_positions.shape[0]
corrmat = np.empty((num_chans, num_chans))

i = n_chunks-1
print('Working on chunk ' + str(i + 1) + ' out of ' + str(n_chunks))
chunk = data[:, i * chunk_size:chunk_size + i * chunk_size]
corrmat[:, :] = get_corrmat(chunk, show_heatmap=False)

#corrmat = np.mean(corrmat, axis=2)
plot_corr_heatmap(corrmat)



i = 0
print('Working on chunk ' + str(i + 1) + ' out of ' + str(n_chunks))
chunk = data[:, i * chunk_size:chunk_size + i * chunk_size]
corrmat[:, :] = get_corrmat(chunk, show_heatmap=False)

#corrmat = np.mean(corrmat, axis=2)
plot_corr_heatmap(corrmat)





data = load_raw_data()
chunk_size = 20000


#data = raw_data[:,9042000:9042000 + chunk_size]
data = rawdata[:,:chunk_size]
corrmatsubset,distmat = get_corrmat(data, show_heatmap=True)




def get_corrmat(data, show_heatmap=False):
    """
    This function computes the correlation matrix between the 255 channels of the probe.

    :param data:
    :param show_heatmap:
    :return:
    """

    channel_positions = get_channel_positions()
    data = data[channel_positions.Numbers]
    #data= data [:, :500000]
    data = highpass_filter_in_chunks(data, 10000)
    #data = highpass(data, F_HIGH = (sampling_freq/2)*0.95, sampleFreq = sampling_freq, passFreq = high_pass_freq)
    #data = data[:15,:]

    #
    #data = highpass(data, F_HIGH = (sampling_freq/2)*0.95, sampleFreq = sampling_freq, passFreq = high_pass_freq)
    #data = lowpass_filter_in_chunks(data, chunk_size=10000)

    n_chans = data.shape[0]
    corrmat = np.empty((n_chans, n_chans), dtype=float)
    #distmat = np.empty((n_chans, n_chans), dtype=float)
    for i in range(n_chans):
        for j in range(n_chans):
            corrmat[i, j] = pearsonr_in_chunks(data[i], data[j])
            #xdiff = channel_positions.Positions[i][0] - channel_positions.Positions[j][0]
            #ydiff = channel_positions.Positions[i][1] - channel_positions.Positions[j][1]
            #distmat[i, j] = np.sqrt(xdiff**2 + ydiff**2)

    if show_heatmap:
        plot_corr_heatmap(corrmat)

    return corrmat
    #return corrmat, distmat


def plot_corr_heatmap(corrmat):
    """
    This function plots a heatmap for correlations between
    :param corrmat:
    :return:
    """
    channel_positions = get_channel_positions()
    f, ax = plt.subplots(figsize=(12, 9))
    hm = sns.heatmap(corrmat,
                     square=True,
                     cmap='RdBu_r'
                     )

    hm.set_xticklabels([])
    hm.set_yticklabels([])

    for i in np.arange(1, corrmat.shape[0], 15):
        ax.axhline(i - 1, color="w",linewidth=1)
        ax.axvline(i - 1, color="w",linewidth=1)
    f.tight_layout()
    return f, ax


def pearsonr_in_chunks(x, y):
    """
    This function computes the pairwise pearson r over the rows of x and y and returns
    the mean correlation coefficient over all rows.

    :param x:
    :param y:
    :return:
    """
    r = pairwise_pearsonr(x, y, axis=0)
    return np.mean(r)  # this is the mean over chunks



def pairwise_pearsonr(x, y, axis=0):
    """
    This function computes row or column wise pearson correlation coefficients
    of x and y

    :param x:
    :param y:
    :param axis:
    :return:
    """
    xm = x - np.mean(x, axis=axis, keepdims=True)
    ym = y - np.mean(y, axis=axis, keepdims=True)
    r_num = np.sum(xm * ym, axis=axis)
    r_den = np.sqrt((xm * xm).sum(axis=axis) * (ym * ym).sum(axis=axis))
    return r_num / r_den


def take_filtered_chunk(start_sample=0, end_sample=30000):
    chunk = low_pass_filter(raw_data.dataMatrix[:, start_sample:end_sample],
                            30000,
                            200,
                            filterType='but',
                            filterOrder=3,
                            filterDirection='twopass')
    return chunk


#Filter for extracellular recording

def highpass(data, BUTTER_ORDER=3, F_HIGH=(sampling_freq / 2) * 0.95,sampleFreq=sampling_freq, passFreq=high_pass_freq):

    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    dims = data.ndim
    axis = 0
    if dims == 2:
        axis = 1
    elif dims == 3:
        axis = 1
    return signal.filtfilt(b, a, data, axis=0)



def highpass_filter_in_chunks(data, chunk_size):
    """
    This function low pass filters large data sizes by cutting them up
    into chunks, and filtering the chunks. It returns the reshaped and
    filtered array

    :param data:
    :param chunk_size:
    :return:
    """
    if data.shape[1] % chunk_size != 0:
        data = data[:, 0:- (data.shape[1] % chunk_size)]

    n_chunks = int(data.shape[1] / chunk_size)
    data = data.reshape((data.shape[0], int(data.shape[1] / n_chunks), n_chunks))

    return highpass(data,  F_HIGH=(sampling_freq / 2) * 0.95, sampleFreq=sampling_freq, passFreq=high_pass_freq)



def low_pass_filter(data, Fsampling, Fcutoff, filterType='but', filterOrder=None, filterDirection='twopass'):
    """
    Low passes the data at the Fcutoff frequency.
    filterType = ´but´ (butterworth) (default) OR ´fir´
    filterOrder = the order of the filter. For the default butterworth filter it is 6
    filterDirection = FilterDirection which defines whether the filter is passed over the data once (and how) or twice
    """
    Wn = np.float32(Fcutoff / (Fsampling / 2.0))
    if filterType == 'fir':
        if filterOrder == None:
            raise ArithmeticError("A filter order is required if the filter is to be a fir and not a but")
        (b, a) = signal.firwin(filterOrder + 1, Wn, width=None, window='hamming', pass_zero=True, scale=True, nyq=1.0)
    else:
        if filterOrder == None:
            filterOrder = 6
        (b, a) = signal.butter(filterOrder, Wn, btype='lowpass', analog=0, output='ba')

    dims = data.ndim
    axis = 0
    if dims == 2:
        axis = 1
    elif dims == 3:
        axis = 1

    if filterDirection == ct.FilterDirection.TWO_PASS:
        filteredData = signal.filtfilt(b, a, data, axis)
    elif filterDirection == ct.FilterDirection.ONE_PASS:
        filteredData = signal.lfilter(b, a, data, axis, zi=None)
    elif filterDirection == ct.FilterDirection.ONE_PASS_REVERSE:
        data = np.fliplr(data)
        filteredData = signal.lfilter(b, a, data, axis, zi=None)
        filteredData = np.fliplr(filteredData)
    return filteredData



def lowpass_filter_in_chunks(data, chunk_size):
    """
    This function low pass filters large data sizes by cutting them up
    into chunks, and filtering the chunks. It returns the reshaped and
    filtered array

    :param data:
    :param chunk_size:
    :return:
    """
    if data.shape[1] % chunk_size != 0:
        data = data[:, 0:- (data.shape[1] % chunk_size)]

    n_chunks = int(data.shape[1] / chunk_size)
    data = data.reshape((data.shape[0], int(data.shape[1] / n_chunks), n_chunks))

    dims = data.ndim
    axis = 0
    if dims == 2:
        axis = 1
    elif dims == 3:
        axis = 1

    return low_pass_filter(data, 30000, 200, filterType='but', filterOrder=3, filterDirection='twopass')


def convertmicrovolts(data):
    return (data - 32768.0) * 0.195




def car(data):
    """
    Common average reference
    :param data:
    :return:
    """
    data_shape = data.shape
    filename = os.path.join(data_folder, 'amplifier_car.bin')
    data_car = np.memmap(filename, dtype='float32', mode='w+', shape=data_shape)

    chunk_size = 500000
    n_chunks = int(data_shape[1] / chunk_size)

    for i in range(n_chunks):
        print("Processing chunk " + str(i + 1) + " of " + str(n_chunks))
        chunk = data[:, i * chunk_size:chunk_size + i * chunk_size]
        chunk[:64] = chunk[:64] - np.median(chunk[:64], axis=0)
        chunk[64:] = chunk[64:] - np.median(chunk[64:], axis=0)
        data_car[:, i * chunk_size:chunk_size + i * chunk_size] = chunk[:]

    # last chunk of different size:
    rest_idx = data_shape[1] % chunk_size
    chunk = data[:, -rest_idx:-1]
    chunk = chunk - np.median(chunk, axis=0)
    data_car[:, -rest_idx+1:] = chunk[:]
    return data_car


def read_sync(syncpath, counterpath):
    sync = np.fromfile(syncpath, dtype=np.uint8)
    syncidx = np.flatnonzero(np.diff(np.int8(sync > 0)) < 0)
    counter = pd.read_csv(counterpath, names=['counter'])
    drops = counter.diff() - 1
    if len(syncidx) != (len(counter) + drops.sum()[0]):
        print
        "WARNING: Number of frames does not match number of sync pulses!"
    matchedpulses = drops.counter.fillna(0).cumsum() + np.arange(len(drops))
    syncidx = syncidx[matchedpulses.astype(np.int32).values]
    syncidx = pd.DataFrame(syncidx, columns=['syncidx'])

    frameidx = np.zeros(sync.shape, dtype=np.int)
    frameidx[0] = -1
    frameidx[syncidx.values] = 1
    frameidx = pd.DataFrame(frameidx, columns=['frame']).cumsum()
    return syncidx, frameidx





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

#test2 w/out repetition of pairs

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



all_electrodes= create_256channels_neuroseeker_prb()



cross= {}
for electrode in channel_positions.Numbers:
    cross[electrode]=[]
    for pair in graph_dict2[electrode]:
        cross[electrode].append(corrmat[electrode, pair])


# distance versus cross

plt.figure()
origin_cmap= plt.get_cmap('hsv')
shrunk_cmap = shiftedColorMap(origin_cmap, start=0.0, midpoint=0.5, stop=1, name='shrunk')
cm=shrunk_cmap
cNorm=colors.Normalize(vmin=0,vmax=15)
scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cm)


symbols = [".",",","_","v","^","<",">","1","2","3","4","8","s","x","+","*","h"]
#colors = ['#c756a4','#2b3081','#27752b','#c04b42','#051006','#d9b08e','#49c2ac','#7e2a46','#c8bf59','#441f5c',
 #         '#c34b67','#b47f3c','#256e67','#dfe7b6','#9fc655']
for i in range(corrmat.shape[0]):
        for j in range(corrmat.shape[0]):
            plt.scatter(distmat[i,j], corrmat[i, j], color=scalarMap.to_rgba(i%15), marker=symbols[j//15],facecolors="None", s=150, linewidths=1)




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
