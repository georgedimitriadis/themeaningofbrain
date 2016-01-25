# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 23:25:00 2014

@author: IntelligentSystem
"""

import os
import video
import vidutils

def savefirstpoke(path):
    subfolders = os.walk(path)
    name = os.path.split(path)[1]
    path = next(subfolders)[0]
    path = next(subfolders)[0]
    basepath = r'C:/Users/IntelligentSystem/Desktop'
    leftrewards = os.path.join(path,'left_rewards.csv')
    videopath = os.path.join(path,'top_video.avi')
    timepath = os.path.join(path,'top_video.csv')
    frontvideo = video.video(videopath,timepath)
    with open(leftrewards) as f:
        timestamp = f.readline()
        pokeframe = frontvideo.frameindex(timestamp)
        vidutils.savemovie(videopath,0,pokeframe-120,pokeframe+120,os.path.join(basepath,name + '.avi'))