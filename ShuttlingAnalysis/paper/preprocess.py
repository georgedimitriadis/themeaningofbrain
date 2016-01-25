# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:47:45 2013

@author: gonca_000
"""

import os
import glob
import shutil
import filecmp
import dateutil
import datetime
import subprocess
import numpy as np
import pandas as pd

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

fronttime_key = 'video/front/time'
frontactivity_key = 'video/front/activity'
fronttrials_key = 'video/front/trials'
toptime_key = 'video/top/time'
whiskertime_key = 'video/whisker/time'
leftpoke_key = 'task/poke/left/activity'
rightpoke_key = 'task/poke/right/activity'
rewards_key = 'task/rewards'
info_key = 'sessioninfo'

max_height_cm = 24.0
max_width_cm = 50.0
height_pixel_to_cm = max_height_cm / 680.0
width_pixel_to_cm = max_width_cm / 1280.0
rail_height_pixels = 100
rail_start_pixels = 200
rail_stop_pixels = 1080
#rail_stop_pixels = 1000
rail_start_cm = rail_start_pixels * width_pixel_to_cm
rail_stop_cm = rail_stop_pixels * width_pixel_to_cm
frames_per_second = 120.0
slipcenter_pixels = [(124, 629), (289, 629), (469, 628), (642, 627),
                     (824, 625), (995, 625), (1161, 624)]
slipcenter_pixels = [(y,x) for x,y in slipcenter_pixels] # crop offset
stepcenter_pixels = [(58, 102), (214, 103), (378, 106), (537, 102),
                     (707, 105), (863, 103), (1026, 97), (1177, 94)]
stepcenter_pixels = [(y+467,x+21) for x,y in stepcenter_pixels] # crop offset
stepcenter_pixels = [(y-50,x) for y,x in stepcenter_pixels]
# small offset for step visualization
stepcenter_cm = [(y*height_pixel_to_cm,x*width_pixel_to_cm)
                 for y,x in stepcenter_pixels]
slipcenter_cm = [(y*height_pixel_to_cm,x*width_pixel_to_cm)
                 for y,x in slipcenter_pixels]

h5filename = 'session.hdf5'
labelh5filename = 'labels.hdf5'
analysisfolder = 'Analysis'
backgroundfolder = 'Background'
playerpath = r'C:\George\Development\Bonsai\bonsai.lesions\Bonsai.Player.exe' #Bonsai.Player.exe is for batch no editor processing, Bonsai.Editor.exe is for editor showing
databasepath = r'C:\Users\IntelligentSystem\kampff.lab@gmail.com\animals\\'

# Example:
# import bs4
# with open(path) as f:
#   markup = f.read()
# bs = bs4.BeautifulSoup(markup,'xml')
# steps,slips = parserois(bs)
def parserois(soup):
    detectors = []
    xdetectors = soup.find_all('Regions')
    for detector in xdetectors:
        rois = []
        xrois = detector.find_all('ArrayOfCvPoint')
        for roi in xrois:
            points = []
            xpoints = roi.find_all('CvPoint')
            for point in xpoints:
                x = int(point.find_all('X')[0].text)
                y = int(point.find_all('Y')[0].text)
                points.append((x,y))
            rois.append(points)
        detectors.append(rois)
    return detectors

def process_subjects(datafolders,  background_generate=False, overwrite=None, visual_analysis=False, database_generate=False):
    for basefolder in datafolders:
        datafolders = [path for path in directorytree(basefolder,1)
                       if os.path.isdir(path)]
        process_sessions(datafolders, background_generate, overwrite, visual_analysis, database_generate)
        
def process_sessions(datafolders, background_generate=False, overwrite=None, visual_analysis=False, database_generate=False):
    
    if background_generate:
        print ('Generating labels...')
        make_sessionlabels(datafolders)
        
        # Check for background files and regenerate if necessary
        for path in datafolders:
            print("Checking backgrounds for " + path + "...")
            backgroundsready = make_backgrounds(path,overwrite)
            if not backgroundsready:
                raise Exception("Aborted due to missing backgrounds!")

    if visual_analysis:
        print("Running analysis pipeline...")
        for path in datafolders:
            make_analysisfolder(path)
            analysispath = os.path.join(path,analysisfolder)
            make_videoanalysis(analysispath)

    if database_generate:
        print("Generating datasets...")
        for i,path in enumerate(datafolders):
            print("Generating dataset for "+ path + "...")
            createdataset(i,path,overwrite=True)
        
def storepath(path):
    return os.path.join(path, analysisfolder, h5filename)
    
def labelpath(path):
    return os.path.join(path, analysisfolder, labelh5filename)

def readtimestamps(path):
    timestamps = pd.read_csv(path,header=None,names=['time'])
    return pd.to_datetime(timestamps['time'])
    
def scaletrajectories(ts,
          sx=width_pixel_to_cm,
          sy=height_pixel_to_cm,
          by=rail_height_pixels,
          my=max_height_cm):
    return [0,my,0,my] - (ts + [0,by,0,by]) * [-sx,sy,-sx,sy]
    
def sliceframe(slices):
    return pd.DataFrame([(s.start,s.stop) for s in slices],
                        columns=['start','stop'])
    
def indexseries(series,index):
    diff = len(index) - len(series)
    if diff > 0:
        msg="WARNING: time series length smaller than index by {0}. Padding..."
        print (str.format(msg,diff))
        lastrow = series.tail(1)
        series = series.append([lastrow] * diff)
    series.index = index
    return series
    
def readdatabase(name):
    path = databasepath + name + '.csv'
    return pd.read_csv(path,
                       header=None,
                       names=['time','event','value'],
                       dtype={'time':pd.datetime,'event':str,'value':str},
                       parse_dates=[0],
                       index_col='time')
    
def readpoke(path):
    return pd.read_csv(path,
                       sep=' ',
                       header=None,
                       names=['activity','time'],
                       dtype={'activity':np.int32,'time':pd.datetime},
                       parse_dates=[1],
                       index_col=1,
                       usecols=[0,1])
                       
def readstep(path,name):
    return pd.read_csv(path,
                       header=None,
                       true_values=['True'],
                       false_values=['False'],
                       names=[name])[name]
        
def createdataset(session,path,overwrite=False):
    h5path = storepath(path)
    if os.path.exists(h5path):
        if overwrite:
            print("Overwriting...")
            os.remove(h5path)
        else:
            print("Skipped!")
            return

    # Load raw data
    fronttime = readtimestamps(os.path.join(path, 'front_video.csv'))
    toptime = readtimestamps(os.path.join(path, 'top_video.csv'))
    leftrewards = readtimestamps(os.path.join(path, 'left_rewards.csv'))
    rightrewards = readtimestamps(os.path.join(path, 'right_rewards.csv'))
    leftpoke = readpoke(os.path.join(path, 'left_poke.csv'))
    rightpoke = readpoke(os.path.join(path, 'right_poke.csv'))
    
    # Load preprocessed data
    trajectorypath = os.path.join(path, analysisfolder, 'trajectories.csv')
    stepactivitypath = os.path.join(path, analysisfolder, 'step_activity.csv')
    gapactivitypath = os.path.join(path, analysisfolder, 'slip_activity.csv')
    trajectories = pd.read_csv(trajectorypath,
                               sep = ' ',
                               header=None,
                               dtype=np.int32,
                               names=['xhead','yhead','xtail','ytail'])
    stepactivity = pd.read_csv(stepactivitypath,
                               sep = ' ',
                               header = None,
                               dtype=np.int32,
                               names=[str.format('stepactivity{0}',i)
                               for i in range(8)])
    gapactivity = pd.read_csv(gapactivitypath,
                              sep = ' ',
                              header = None,
                              dtype=np.int32,
                              names=[str.format('gapactivity{0}',i)
                              for i in range(7)])
    trajectories = indexseries(trajectories,fronttime)
    scaledtrajectories = scaletrajectories(trajectories[trajectories >= 0])
    trajectories[trajectories < 0] = np.NaN
    trajectories[trajectories >= 0] = scaledtrajectories
    stepactivity = indexseries(stepactivity,fronttime)
    gapactivity = indexseries(gapactivity,fronttime)
    
    # Compute speed
    speed = trajectories.diff()
    timedelta = pd.DataFrame(fronttime.diff() / np.timedelta64(1,'s'))
    timedelta.index = speed.index
    speed = pd.concat([speed,timedelta],axis=1)
    speed = speed.div(speed.time,axis='index')
    speed.columns = ['xhead_speed',
                     'yhead_speed',
                     'xtail_speed',
                     'ytail_speed',
                     'timedelta']
    speed['timedelta'] = timedelta
    
    # Compute reward times
    leftrewards = pd.DataFrame(leftrewards)
    rightrewards = pd.DataFrame(rightrewards)
    leftrewards['side'] = 'left'
    rightrewards['side'] = 'right'
    rewards = pd.concat([leftrewards,rightrewards])
    rewards.sort(columns=['time'],inplace=True)
    rewards.reset_index(drop=True,inplace=True)
    
    # Compute trial indices and environment state
    trialindex = pd.concat([fronttime[0:1],rewards.time])
    trialseries = pd.Series(range(len(trialindex)),
                            dtype=np.int32,
                            name='trial')
    if len(trialindex) > 200:
        print("WARNING: Trial count exceeded 200!")

    steppath = os.path.join(path, 'step{0}_trials.csv')
    axisname = 'stepstate{0}'
    stepstates = [readstep(str.format(steppath,i),str.format(axisname,i)) for i in range(1,7)]
    trialseries = pd.concat([trialseries] + stepstates,axis=1)
    trialseries.fillna(method='ffill',inplace=True)
    trialseries = trialseries[0:len(trialindex)]
    trialseries.index = trialindex
    trialseries = trialseries.reindex(fronttime,method='ffill')

    # Find frames for start and end of each trial
    frames_start = [i for i,x in enumerate(trajectories.xhead) if not np.isnan(x) and np.isnan(trajectories.xhead[i-1])]
    frames_end = [i for i,x in enumerate(trajectories.xhead) if not np.isnan(x) and np.isnan(trajectories.xhead[i+1])]
    frames_start_times = fronttime[frames_start]
    frames_end_times = fronttime[frames_end]

    frames_indexes = np.searchsorted(trialindex.values, frames_end_times.values)
    frames_indexes = np.append(frames_indexes, frames_indexes[-1])
    frames_indexes = [i for i,x in enumerate(frames_indexes[:-1]) if x != frames_indexes[i+1]]

    frames_start = [frames_start[i] for i in frames_indexes]
    frames_end = [frames_end[i] for i in frames_indexes]
    frames_start_times = [frames_start_times.iloc[i] for i in frames_indexes]
    frames_end_times = [frames_end_times.iloc[i] for i in frames_indexes]

    trials_durations = np.array(frames_end_times) - np.array(frames_start_times)

    data = list(map(list, zip(*[frames_start, frames_end, frames_start_times, frames_end_times, trials_durations])))
    trials_start_stop_info = pd.DataFrame(data=data,
                                          columns=['start frame', 'end frame', 'start frame time', 'end frame time', 'trial duration'])
    
    # Generate session info
    starttime = fronttime[0].replace(second=0, microsecond=0)
    dirname = os.path.basename(path)
    subjectfolder = os.path.dirname(path)
    subject = os.path.basename(subjectfolder)
    protocol = sessionlabel(path)
    database = readdatabase(subject)
    gender = str.lower(database[database.event == 'Gender'].ix[0].value)
    birth = database[database.event == 'Birth']
    age = starttime - birth.index[0]
    weights = database[(database.event == 'Weight') &
                       (database.index < starttime)]
    weight = float(weights.ix[weights.index[-1]].value)
    housed = database.event == 'Housed'
    lefthistology = database.event == 'Histology\LesionLeft'
    righthistology = database.event == 'Histology\LesionRight'
    cagemate = database[housed].ix[0].value if housed.any() else 'None'
    lesionleft = float(database[lefthistology].value if lefthistology.any() else 0)
    lesionright = float(database[righthistology].value if righthistology.any() else 0)
    watertimes = database[(database.event == 'WaterDeprivation') &
                          (database.index < starttime)]
    if len(watertimes) > 0:
        deprivation = starttime - watertimes.index[-1]
    else:
        deprivation = 0
    info = pd.DataFrame([[subject,session,dirname,starttime,protocol,gender,age,
                          weight,deprivation,lesionleft,lesionright,cagemate]],
                        columns=['subject',
                                 'session',
                                 'dirname',
                                 'starttime',
                                 'protocol',
                                 'gender',
                                 'age',
                                 'weight',
                                 'deprivation',
                                 'lesionleft',
                                 'lesionright',
                                 'cagemate'])
    info.set_index(['subject','session'],inplace=True)
    
    # Generate big data table
    frame = pd.Series(range(len(fronttime)),dtype=np.int32,name='frame')
    frame = indexseries(frame,fronttime)
    frontactivity = pd.concat([frame,
                               trialseries,
                               trajectories,
                               speed,
                               stepactivity,
                               gapactivity],
                               axis=1)

    trials_start_stop_info.to_hdf(h5path, fronttrials_key)
    fronttime.to_hdf(h5path, fronttime_key)
    frontactivity.to_hdf(h5path, frontactivity_key)
    toptime.to_hdf(h5path, toptime_key)
    leftpoke.to_hdf(h5path, leftpoke_key)
    rightpoke.to_hdf(h5path, rightpoke_key)
    rewards.to_hdf(h5path, rewards_key)
    info.to_hdf(h5path, info_key)
    
    # Optional whisker camera info
    whiskerpath = os.path.join(path, 'whisker_video.csv')
    if os.path.exists(whiskerpath):
        whiskertime = readtimestamps(whiskerpath)
        whiskertime.to_hdf(h5path, whiskertime_key)

def sessionlabel(path):
    protocolfilefolder = os.path.join(dname,'../protocolfiles/lesionsham')
    trialfiles = [f for f in glob.glob(path + r'\step*_trials.csv')]
    for folder in os.listdir(protocolfilefolder):
        match = True
        targetfolder = os.path.join(protocolfilefolder,folder)
        for f1,f2 in zip(trialfiles,os.listdir(targetfolder)):
            targetfile = os.path.join(targetfolder,f2)
            if not filecmp.cmp(f1,targetfile):
                match = False
                break
        
        if match:
            return folder
    return 'stable'

def make_sessionlabels(datafolders):
    for path in datafolders:
        label = sessionlabel(path)
        session_labels_file = os.path.join(path,'session_labels.csv')
        if not os.path.exists(session_labels_file):
            np.savetxt(session_labels_file,[['protocol',label]],delimiter=':',fmt='%s')
            
def make_analysisfolder(path):
    analysispath = os.path.join(path,analysisfolder)
    if not os.path.exists(analysispath):
        print("Creating analysis folder in "+path)
        os.mkdir(analysispath)

def make_backgrounds(path=None,overwrite=None):
    currdir = os.getcwd()
    if path is not None:
        os.chdir(path)
    
    if os.path.exists(backgroundfolder):
        if overwrite is None:
            overwrite = input('Backgrounds exist - overwrite? (y/n)')
        if overwrite != 'y':
            return True
        else:
            shutil.rmtree('Background')
        
    backgroundbuilder = os.path.join(dname, r'bonsai/background_builder.bonsai')
    print("Extracting raw backgrounds...")
    subprocess.call([playerpath, backgroundbuilder])
    os.chdir(currdir)
    return True
    
def make_videoanalysis(path):
    if not os.path.exists(path):
        return
    
    global dname
    currdir = os.getcwd()
    print("Processing video of "+ path + "...")
    os.chdir(path)
    
    if not os.path.exists('trajectories.csv'):
        videoprocessing = os.path.join(dname, r'bonsai/video_preprocessor_loadcells.bonsai')
        print("Analysing video frames playerpath: "+playerpath+", videoprocessing: "+videoprocessing)
        subprocess.call([playerpath, videoprocessing])
        
    videotimepath = 'videotime.csv'
    if not os.path.exists(videotimepath):
        frametimespath = os.path.join(path, '../front_video.csv')
        frametimes = np.genfromtxt(frametimespath,dtype=str)
        print("Generating relative frame times...")
        datetimes = [dateutil.parser.parse(timestr) for timestr in frametimes]
        videotime = [(time - datetimes[0]).total_seconds() for time in datetimes]    
        np.savetxt(videotimepath, np.array(videotime), fmt='%s')

    os.chdir(currdir)
    
def directorytree(path,level):
    if level > 0:
        return [directorytree(path + '\\' + name, level-1) for name in os.listdir(path)]
    return path