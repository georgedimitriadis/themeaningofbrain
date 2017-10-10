# -*- coding: utf-8 -*-
"""
Created on Tue Mar 03 16:13:49 2015

@author: kampff
"""


import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy
#from sklearn.decomposition import FastICA
from mpl_toolkits.mplot3d import Axes3D


# Distance from 128channels

# all_electrodes
r1 = np.array([103,	101, 99,	97,	95,	93,	91,	89,	87,	70,	66,	84,	82,	80,	108, 110, 47, 45, 43, 41, 1, 61, 57,
                   36, 34, 32, 30, 28, 26, 24, 22, 20])
r2 = np.array([106, 104, 115, 117, 119, 121, 123, 125, 127, 71, 67, 74, 76, 78, 114, 112,
                   49, 51, 53, 55, 2, 62, 58, 4, 6, 8, 10, 12, 14, 21, 19, 16])
r3 = np.array([102, 100, 98, 96, 94, 92, 90, 88, 86, 72, 68, 65, 85, 83, 81, 111, 46, 44, 42, 40, 38, 63, 59,
                   39, 37, 35, 33, 31, 29, 27, 25, 23])
r4 = np.array([109, 107, 105, 116, 118, 120, 122, 124, 126, 73, 69, 64, 75, 77, 79, 113,
                   48, 50, 52, 54, 56, 0, 60, 3, 5, 7, 9, 11, 13, 15, 18,-1])

all_electrodes_concat = np.concatenate((r1, r2, r3, r4))
all_electrodes = all_electrodes_concat.reshape((4, 32))
all_electrodes = np.flipud(all_electrodes.T)

# orderedsites
electrode_coordinate_grid = list(itertools.product(np.arange(1, 33), np.arange(1, 5)))
orderedSites = electrode_coordinate_grid

# sitespositions
scaleX=25
scaleY=25
scale = np.array([scaleX,scaleY])
probeConfiguration=np.zeros((32,4,2))
for i in range(32):
    for j in range(4):
        probeConfiguration[i,j]= np.array([i,j])

probeConfiguration = np.flipud(probeConfiguration) # the (0,0) for Y,Z is the elect (31,0) left bottom
probeConfiguration = probeConfiguration * scale
sitesPosition = probeConfiguration

def eval_dist(referencecoordinates, IVM, juxta):
    '''
    referenceSite = site to which the IVM coordinates refer to
    IVM = coordinates of the reference site on the probe
    juxta = juxta probe coordinates
    sitesPositionsFN = path to sites positions file.
    '''

    scaleX=25
    scaleY=25
    scale = np.array([scaleX,scaleY])
    probeConfiguration=np.zeros((32,4,2))
    for i in range(32):
        for j in range(4):
            probeConfiguration[i,j]= np.array([i,j])

    probeConfiguration = np.flipud(probeConfiguration) # the (0,0) for Y,Z is the elect (31,0) left bottom
    probeConfiguration = probeConfiguration * scale
    sitesPosition = probeConfiguration

    electrode_coordinate_grid = list(itertools.product(np.arange(1, 33), np.arange(1, 5)))
    orderedSites = electrode_coordinate_grid

    referencePosition = np.zeros(2)
    referencePosition[0] = referencecoordinates[0]
    referencePosition[1] = referencecoordinates[1]

    #orderedSites = id_rever
    NumSites =  len(orderedSites)


    for i in range(32):
         for j in range(4):
            sitesPosition[i][j][0]= sitesPosition[i][j][0] - referencePosition[0]
            sitesPosition[i][j][1]= sitesPosition[i][j][1] - referencePosition[1]



    newPositions = np.zeros((128, 3))

    #newPositions[:, 1] = np.copy(sitesPositions[:, 0])
    #newPositions[:, 2] = np.copy(sitesPositions[:, 1])
    newPositions = np.zeros((NumSites, 3))
    newPositions[:, 2] = np.copy(sitesPosition[:,:,0].reshape(1,128))
    newPositions[:, 1] = np.copy(sitesPosition[:,:,1].reshape(1,128))

    for i in range(NumSites):
        newPositions[i, 0] = newPositions[i, 2] * np.cos(0.8412486985)  # 48.2º

    for i in range(NumSites):
        newPositions[i, 2] = newPositions[i, 2] * np.sin(0.8412486985)  # 48.2º

    spikesPositions = np.zeros((NumSites, 3))
    spikesDistances = np.zeros(NumSites)

    spikesPositions = np.copy(newPositions)
    for i in range(NumSites):
        spikesPositions[i] = spikesPositions[i] - IVM + juxta

    for j in range(NumSites):
        spikesDistances[j] = np.sqrt(spikesPositions[j][0]**2 +
                                     spikesPositions[j][1]**2 +
                                     spikesPositions[j][2]**2)

    return spikesPositions, spikesDistances

#How to get the distances values?

cellname = '2015_09_04_pair7.1'

# extraPos and juxtaPos have to be numpy arrays
# I had to add a decimal place in all axes
# The Z axis has to be flipped





150.6720766	-41.59289885	-4426.736869	339.6	-12.8	-4307.8


extraPos = np.array([150.6720766,	-41.59289885,	-4426.736869])*np.array([1, 1, -1])
juxtaPos = np.array([339.6,	-12.8,	-4307.8])*np.array([1, 1, -1])

referencecoordinates = (-310, 37.5) #ref point in extra probe is the Tip (z,y)
#referencecoordinates = (0, 37.5) #ref point in extra probe is bottom center (z,y)
refSite = referencecoordinates

pos, dist = eval_dist(refSite, extraPos, juxtaPos)


min_dist = dist.min()
channel_min_dist = electrode_coordinate_grid[dist.argmin()]
channel_intan = all_electrodes[(channel_min_dist[0])-1][(channel_min_dist[1])-1]
print(min_dist)
#print(channel_min_dist)
print(channel_intan)

# Save distances and positions for the cell

analysis_folder ='D:\\Protocols\\PairedRecordings\\Neuroseeker128\\Data\\2015-08-21\\Analysis'

i=1
np.save(os.path.join(analysis_folder,'distances_Cell'+ good_cells[i] + '.npy'), dist)
np.save(os.path.join(analysis_folder,'positions_Cell'+ good_cells[i] + '.npy'), pos)


# Schematic of the relative positions of the juxtacellular probe and the
# electrodes of the silicon probe ()
fig, ax = plt.subplots()
ax.scatter(pos[:, 0], pos[:, 2], color='b')
ax.scatter(0, 0, color='r')
ax.set_title('XoZ plane\n'+cellname, fontsize=20)
ax.set_aspect('equal')
plt.draw()
plt.show()

####scheme of heatmap

A = np.copy(dist)

orderedSites = all_electrodes.reshape(1,128)
Amod = np.reshape(A,(32,4))
fig, ax = plt.subplots()
#ax.set_title('Distances Heatmap',fontsize=40, position=(0.8,1.02))
plt.axis('off')
im = ax.imshow(Amod, cmap=plt.get_cmap('jet'),vmin = np.min(A),vmax= np.max(A))
cb = fig.colorbar(im, ticks = [np.min(A), 0,np.max(A)])
cb.ax.tick_params(labelsize = 20)
plt.show()

####3D

spikesPositions =pos
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(spikesPositions[:,0],spikesPositions[:,1],spikesPositions[:,2])
ax.scatter(0,0,0,c='g')





# Distances from 32channels

orderedSites = np.genfromtxt('E:\Data\orderedSites.dat', delimiter=' ',dtype="i8")

def eval_dist(referenceSite, IVM, juxta, sitesPositionsFN='E:\Data\probeconfiguration.dat'):
    '''
    referenceSite = site to which the IVM coordinates refer to
    IVM = coordinates of the reference site on the probe
    juxta = juxta probe coordinates
    sitesPositionsFN = path to sites positions file.
    '''

    sitesPositions = np.genfromtxt(sitesPositionsFN, delimiter=' ')
    NumSites = sitesPositions.shape[0]

    referencePosition = np.zeros(2)
    referencePosition[0] = sitesPositions[referenceSite][0]
    referencePosition[1] = sitesPositions[referenceSite][1]

    for i in range(NumSites):
        for j in range(2):
            sitesPositions[i, j] = sitesPositions[i, j] - referencePosition[j]

    newPositions = np.zeros((NumSites, 3))
    newPositions[:, 1] = np.copy(sitesPositions[:, 0])
    newPositions[:, 2] = np.copy(sitesPositions[:, 1])

    for i in range(NumSites):
        newPositions[i, 0] = newPositions[i, 2] * np.cos(0.8412486985)  # 48.2º

    for i in range(NumSites):
        newPositions[i, 2] = newPositions[i, 2] * np.sin(0.8412486985)  # 48.2º

    spikesPositions = np.zeros((NumSites, 3))
    spikesDistances = np.zeros(NumSites)

    spikesPositions = np.copy(newPositions)
    for i in range(NumSites):
        spikesPositions[i] = spikesPositions[i] - IVM + juxta

    for j in range(NumSites):
        spikesDistances[j] = np.sqrt(spikesPositions[j][0]**2 +
                                     spikesPositions[j][1]**2 +
                                     spikesPositions[j][2]**2)

    return spikesPositions, spikesDistances

######################
cellname = '2014_11_13_pair7.0'
# extraPos and juxtaPos have to be numpy arrays
# I had to add a decimal place in all axes
# The Z axis has to be flipped


-163.2410881	-194.9309165	-3298.073118
	252	104.7	-2397.8

extraPos = np.array([-334.6634074,	99.71486382,	-2467.206884	])*np.array([1, 1, -1])
juxtaPos = np.array([	252,	104.7,	-2397.8])*np.array([1, 1, -1])
refSite = 17


pos, dist = eval_dist(refSite, extraPos, juxtaPos)

min_dist = dist.min()
channel_min_dist = dist.argmin()
print(min_dist)
print(channel_min_dist)

# Schematic of the relative positions of the juxtacellular probe and the
# electrodes of the silicon probe ()
fig, ax = plt.subplots()
ax.scatter(pos[:, 0], pos[:, 2], color='b')
ax.scatter(0, 0, color='r')
ax.set_title('XoZ plane\n'+cellname, fontsize=20)
ax.set_aspect('equal')
plt.draw()
plt.show()

####3D

spikesPositions =pos
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(spikesPositions[:,0],spikesPositions[:,1],spikesPositions[:,2])
ax.scatter(0,0,0,c='g')

####Plot distance versus amplitude p2p per channel

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')



p2pfilename=r'E:\Data\2014-10-17\Analysis\Analysis_100Hz_4ms\p2p_EXTRA_Cell1.npy'
stdvp2pfilename= r'E:\Data\2014-10-17\Analysis\Analysis_100Hz_4ms\error_EXTRA_Cell1.npy'

distancesfilename= r'E:\Data\2014-10-17\Analysis\Analysis_100Hz_4ms\distances_EXTRA_Cell1.npy'


p2p = np.load(p2pfilename) # from ch0 to ch31
stdvp2p = np.load(stdvp2pfilename)

distances = np.load(distancesfilename)# from ch0 to ch31
stdvdistances = 30


channel_order= [22, 2, 29, 9,3,28,23,13,18,8,4,27,17,12,19,14,5,26,16,11,20,15,6,25,30,10,21,1,7,24,31,0]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(p2p[channel_order], '-', label = 'P2P', color='m',linewidth=2.0)
ax.fill_between(np.arange(32), p2p[channel_order]-stdvp2p[channel_order], p2p[channel_order]+stdvp2p[channel_order], color = 'm', alpha=0.5 )
ax2 = ax.twinx()
ax2.plot(distances[channel_order], '-', label = 'Distance', color='b',linewidth=2.0)
ax2.fill_between(np.arange(32), distances[channel_order] - 30.0, distances[channel_order] + 30.0,color = 'b', alpha=0.2 )
ax.grid()
ax.set_xlabel('Channel number', fontsize=20)
ax.set_ylabel('Peak-Peak Amplitude (\u00B5V)', fontsize=20)
ax2.set_ylabel('Distance (\u00B5m)', fontsize=20)
ax.set_ylim(0, 40)
ax2.set_ylim(0, 300)
ax.set_xlim(0, 31)
plt.xticks(range(32), [channel_order[i] for i in np.arange(0,len(channel_order))], fontsize=15)
plt.show()
