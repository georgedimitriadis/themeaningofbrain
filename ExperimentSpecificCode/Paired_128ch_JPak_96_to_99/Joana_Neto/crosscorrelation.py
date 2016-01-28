
import numpy as np
import h5py


import numpy as np
import h5py

def crosscorrelate(sua1, sua2, lag=None, n_pred=1, predictor=None,
                   display=False, kwargs={}):


    #calculate cross differences in spike times
    differences = np.array([])
    pred = np.array([])
    for k in xrange(0, sua1.size):
        differences = np.append(differences, sua1[k] - sua2[np.nonzero(
            (sua2 > sua1[k] - lag) & (sua2 < sua1[k] + lag))])
    if predictor == 'shuffle':
        for k in xrange(0, sua1.size):
            pred = np.append(pred, sua1[k] - sua2_[np.nonzero(
                (sua2_ > sua1[k] - lag) & (sua2_ < sua1[k] + lag))])
    if reverse is True:
        differences = -differences
        pred = -pred
    norm = np.sqrt(sua1.size * sua2.size)
    return differences, pred, norm





def open_hdf5_kwx(filename):

    f = h5py.File(filename,'r')
    dataset = f['/channel_groups/0/features_masks']
    #dataset = f['/channel_groups/1/features_masks']
    dataset.shape
    teste= dataset[:,:,1]
    gh=teste[:,slice(0,96,3)]
    return np.array(gh)



def open_hdf5_kw(filename):
    f = h5py.File(filename,'r')
    times = f['/channel_groups/0/spikes/time_samples']
#    times = f['/channel_groups/1/spikes/time_samples']
    return np.array(times)


def times_spikes_ch(maskdata,timedata):
    global time_ch

    time_ch=[]
    for i in np.arange(32):
        maskdata_ch=maskdata[:,i]
        t=[timedata[x] for x in range(np.shape(timedata)[0]) if maskdata_ch[x]> 0]
        time_ch.append(t)
        i=i+1


maskdata= open_hdf5_kwx(r'E:\Data\2014-03-26\Klusta_spikes\amplifier_depth_90um_s16.kwx')
timedata= open_hdf5_kw(r'E:\Data\2014-03-26\Klusta_spikes\amplifier_depth_90um_s16.kwik')
times_spikes_ch(maskdata,timedata)


juxta_times= all_cells_spike_triggers['2']

def plot_crosscorrelation(juxta, timematriz):

    cross = []
    juxta_spikes_ms = np.asarray(juxta /30, dtype=np.float64)
    for i in np.arange(32):
        channel= np.array(timematriz[i])/30
        cross_array= crosscorrelate( channel,juxta_spikes_ms, lag=50)
        cross.append(cross_array)

    sites_order_geometry= [0,31,24,7,1,21,10,30,25,6,15,20,11,16,26,5,14,19,12,17,27,4,8,18,13,23,28,3,9,29,2,22]
    subplot_order = [2,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65]
    for i in np.arange(32):
        plt.subplot(22,3,subplot_order [i])
        plt.hist(cross[sites_order_geometry[i]][0],bins=100)












    subplot(22,3,2) #channel 0
    plt.hist(cross[0][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)


    subplot(22,3,5)#channel 31
    plt.hist(cross[31][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,7)#channel 24
    plt.hist(cross[24][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,9)#channel7
    plt.hist(cross[7][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)


    subplot(22,3,11)#1
    plt.hist(cross[1][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,13)#channel21
    plt.hist(cross[21][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,15)#channel10
    plt.hist(cross[10][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,17)#channel30
    plt.hist(cross[30][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,19)#channel25
    plt.hist(cross[25][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,21)#channel6
    plt.hist(cross[6][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,23)#channel15
    plt.hist(cross[15][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,25)#channel20
    plt.hist(cross[20][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,27)#channel11
    plt.hist(cross[11][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,29)#channel16
    plt.hist(cross[16][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,31)#channel26
    plt.hist(cross[26][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,33)#channel5
    plt.hist(cross[5][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,35)#channel14
    plt.hist(cross[14][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)


    subplot(22,3,37)#channel19
    plt.hist(cross[19][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,39)#channel12
    plt.hist(cross[12][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,41)#channel17
    plt.hist(cross[17][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,43)#channel27
    plt.hist(cross[27][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,45)#channel4
    plt.hist(cross[4][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,47)#channel8
    plt.hist(cross[8][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,49)#channel18
    plt.hist(cross[18][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,51)#channel13
    plt.hist(cross[13][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,53)#channel23
    plt.hist(cross[23][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,55)#channel28
    plt.hist(cross[28][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,57)#channel3
    plt.hist(cross[3][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,59)#channel9
    plt.hist(cross[9][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,61)#channel29
    plt.hist(cross[29][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,63)#channel2
    plt.hist(cross[2][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)

    subplot(22,3,65)#channel22
    plt.hist(cross[22][0],bins=100)
    ylim(0, ysup)
    xlim(-x,x)


plt.figure(); plot_crosscorrelation(juxta_times,time_ch)





        #kernel=stats.gaussian_kde(time_ch[sites_order_geometry[i]],bw_method='scott')
        #plt.plot(kernel(range(0,1000)),label='h='+str(np.round(kernel.factor,decimals=2)))
