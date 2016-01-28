# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 19:29:08 2014

@author: IntelligentSystems
"""

import os
import glob
import shuttling
import numpy as np
from dateutil import parser as dateparser
from datetime import timedelta

def genfromsessions(folders, basemodule=shuttling):
    result = []
    for path in folders:
        print "Processing {0}...".format(path)
        result.append(basemodule.genfromtxt(path))
    return result
    
def genfromsubjects(folders, days=None, basemodule=shuttling):
    result = []
    for path in folders:
        print "Processing {0}...".format(path)
        result.append(genfromsessions(findsessions(path, days), basemodule))
    return result
    
def getsessiondatetime(folder):
    datetimestr = os.path.split(folder)[1].split('-')
    datestr = datetimestr[0].replace('_','-')
    timestr = datetimestr[1].replace('_',':')
    return dateparser.parse(str.join('-',[datestr,timestr]))
    
def findsessions(folder, days=None):
    sessionpaths = glob.glob(os.path.join(folder,'**/front_video.csv'))
    folders = [os.path.split(path)[0] for path in sessionpaths]
    
    def groupsessions(folders):
        folderdates = [getsessiondatetime(sf) for sf in np.sort(folders)]
        isis = np.insert(np.diff(folderdates), 0, timedelta.max)
        groups = []
        group = None
        for isi,sf in zip(isis,folders):
            if group is None or isi > timedelta(hours = 10):
                group = []
                groups.append(group)
            group.append(sf)
        return groups
    
    groups = groupsessions(folders)
    if days is not None:
        groups = [groups[day] for day in days]
    
    return [item for group in groups for item in group]