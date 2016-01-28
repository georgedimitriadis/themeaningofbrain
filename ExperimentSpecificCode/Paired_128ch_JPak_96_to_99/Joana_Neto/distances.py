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
from sklearn.decomposition import FastICA
from mpl_toolkits.mplot3d import Axes3D

#ALL_ELECTRODES
r1 = np.array([103,	101, 99,	97,	95,	93,	91,	89,	87,	70,	66,	84,	82,	80,	108,	110,	47,	45,	43,	41,	1, 61,	57,
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



#####ORDEREDSITES
electrode_coordinate_grid = list(itertools.product(np.arange(1, 33), np.arange(1, 5)))
orderedSites = electrode_coordinate_grid


#####SITESPOSITION
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



###REFERENCE POINT EXTRACELLULAR

referencecoordinates = (0, 37.5) #ref point bottom center (z,y)
referencecoordinates = (-310, 37.5) #ref point Tip (z,y)





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

######################
cellname = 'cell4_0'



cell7''(196.5;0;-4865)IVM(408.7;0;-5282.4)

cell7(204;0;-4850)IVM(408.7;0;-5282.4)

#cell5(331.5 0 -4594.8)IVM(494.5399 0 -4920.042)

# extraPos and juxtaPos have to be numpy arrays
# I had to add a decimal place in all axes
# The Z axis has to be flipped
cell4(-185;y;-4094)IVM(313.3;y;-4640)
 IVM 468.9159 33.3 -4695.594 Patch 437.4 33.4 -4462.1

 IVM 468.9159 33.3 -4695.594 Patch 437.4 33.4 -4462.1


494.5399 0 -4920.042	161.4 47.2 -4415.2

extraPos = np.array([494.5399, 0, -4920.042 ])*np.array([1, 1, -1])
juxtaPos = np.array([161.4, 47.2, -4415.2 ])*np.array([1, 1, -1])
#referencecoordinates = (-310, 37.5) #ref point Tip (z,y)
referencecoordinates = (0, 37.5) #ref point bottom center (z,y)


referencecoordinates = (0, 37.5)
refSite = referencecoordinates

pos, dist = eval_dist(refSite, extraPos, juxtaPos)


min_dist = dist.min()
channel_min_dist = electrode_coordinate_grid[dist.argmin()]
#channel_intan = orderedSites[0, dist.argmin()]
print(min_dist)
print(channel_min_dist)
#print(channel_intan)

i=3

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


###average
###first load data 3D and average

def heatmapp_amplituide(all_cells_ivm_filtered_data, good_cells_number = 0):

    voltage_step_size = 0.195e-6
    extra_average_V = np.average(all_cells_ivm_filtered_data[good_cells[good_cells_number]][:,:,:],axis=2) * voltage_step_size
    extra_average_microVolts = extra_average_V * 1000000
    orderedSites = all_electrodes.reshape(1,128)
    amplitude = np.zeros(128)
    for j in np.arange(128):
        amplitude[j] = abs(np.min(extra_average_microVolts[orderedSites[0][j],:])) + abs(np.max(extra_average_microVolts[orderedSites[0][j],:]))
    return amplitude

i = 0
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

