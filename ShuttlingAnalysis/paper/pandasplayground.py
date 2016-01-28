# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 19:53:20 2014

@author: Gonçalo
"""

import cv2
import video
import siphon
import imgproc
import preprocess
import numpy as np
import pandas as pd
import activitytables
import activitymovies
import matplotlib.pyplot as plt

days = range(1,5)
stepcenters = [(58, 102), (214, 103), (378, 106), (537, 102),
               (707, 105), (863, 103), (1026, 97), (1177, 94)]
stepcenters = [(y+467,x+21) for x,y in stepcenters]
stepcenters = [(y-50,x) for y,x in stepcenters]
datafolders = [r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_20',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_21',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_22',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_23',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_24',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_25',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_26',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_27',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_28',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_29',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_36',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_37',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_38',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_39',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_48',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_49',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_50',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_51',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_52',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_53',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_54',
               r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_55']
cr = activitytables.read_subjects(datafolders,days=days,
                                  selector=lambda x:activitytables.crossings(x,True,False))
rr = activitytables.read_subjects(datafolders,days=days,
                                  key=activitytables.rewards_key)
info = activitytables.read_subjects(datafolders,days=days,
                                    key=activitytables.info_key)
                                    
# For video analysis (4,10,12,16)
days = [0]
datafolders = [r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_21']
cr = activitytables.read_subjects(datafolders,days=days)
rr = activitytables.read_subjects(datafolders,days=days,key=activitytables.rewards_key)
info = activitytables.read_subjects(datafolders,days=days,key=activitytables.info_key)


def clusterstepframes(cr,info,leftstep,rightstep):
    # Compute step times
    stepactivity = cr.iloc[:,16:24]
    stepdiff = stepactivity.diff()
    steppeaks = siphon.findpeaksMax(stepdiff,1500)
    pksloc = [[stepdiff.index.get_loc(peak) for peak in step] for step in steppeaks]    
    
    # Tile step frames
    vidpaths = activitymovies.getmoviepath(info)
    timepaths = activitymovies.gettimepath(info)
    backpaths = activitymovies.getbackgroundpath(info)
    videos = [video.video(path,timepath) for path,timepath in zip(vidpaths,timepaths)]

    def getstepframes(stepindex,flip=False):
        stepcenterxcm = stepcenters[stepindex][1] * preprocess.width_pixel_to_cm
        framehead = [p for p in pksloc[stepindex]
                     if (cr.xhead[p] < stepcenterxcm if not flip else cr.xhead[p] > stepcenterxcm)]
        
        frames = [imgproc.croprect(stepcenters[stepindex],(200,200),videos[0].frame(p))
                  for p in framehead]
        backgrounds = [imgproc.croprect(stepcenters[stepindex],(200,200),activitymovies.getbackground(backpaths[0],videos[0].timestamps[p]))
                       for p in framehead]
        frames = [cv2.subtract(f,b) for f,b in zip(frames,backgrounds)]
        if flip:
            frames = [cv2.flip(f,1) for f in frames]
        return frames,framehead

    leftframes,leftindices = getstepframes(leftstep,False)
    rightframes,rightindices = getstepframes(rightstep,True)
    frames = np.array(leftframes + rightframes)
    frameindices = np.array(leftindices + rightindices)
    sortindices = np.argsort(frameindices)
    frames = frames[sortindices]
    frameindices = frameindices[sortindices]
    
    R,labels,h = imgproc.cluster(frames,videos[0],frameindices)
    return frames,stepdiff,steppeaks,pksloc,R,labels
    
frames,stepdiff,steppeaks,pksloc,R,labels = clusterstepframes(cr,info,4,3)

# Plot step activity and step candidates
ax = plt.gca()
ax.plot(stepdiff)
ax.set_color_cycle(None)
[ax.plot(l,stepdiff.loc[p].ix[:,i],'.') for i,(l,p) in enumerate(zip(pksloc,steppeaks))]
plt.draw()

ax.plot(stepdiff.stepactivity3)
ax.plot(pksloc[3],stepdiff.loc[steppeaks[3]].ix[:,3],'r.')
plt.draw()

# Raw frame tiles
#frames = [f if x < stepcenterxcm else cv2.flip(f,1) for f,x in zip(frames,framehead)]
#labels = np.array([str(p) for p in pksloc[3]])
#tiles = imgproc.tile(frames,6,3)
#activitymovies.showmovie(tiles)

# Showing a video clip
#video.showmovie(videos[0],93571)
                              
# Compute time to next reward (including first trial)
rrdiff = rr.groupby(level=[0,1]).diff()
nulldiff = rrdiff.time.isnull()
firstrr = rr.time[nulldiff] - info.starttime
rrdiff.time[nulldiff] = firstrr
rrsec = rrdiff.time.map(lambda x:x / np.timedelta64(1, 's'))

rrdata = rrsec.groupby(level=[0,1]).mean()
rryerr = rrsec.groupby(level=[0,1]).std()
rrgdata = activitytables.groupbylesionvolumes(pd.concat([rrdata,rryerr],axis=1),info)

data = cr.groupby(level=[0,1])['duration'].mean()
yerr = cr.groupby(level=[0,1])['duration'].std()
lesionvolume = info['lesionleft'] + info['lesionright']
lesionvolume.name = 'lesionvolume'
g = pd.concat([data,yerr,lesionvolume,info['cagemate']],axis=1)
lesionorder = g[g['lesionvolume'] > 0].sort('lesionvolume',ascending=False)
controls = lesionorder.groupby('cagemate',sort=False).median().index
controlorder = g.reset_index().set_index('subject').ix[controls]
controlorder.set_index('session',append=True,inplace=True)
result = pd.concat([controlorder,lesionorder])
result['lesion'] = ['lesion' if v > 0 else 'control'
                    for v in result['lesionvolume']]
result.reset_index(inplace=True)
result.sort(['session','lesion'],inplace=True)
result.set_index(['session','lesion','subject'],inplace=True)
result.drop(['lesionvolume','cagemate'],axis=1,inplace=True)