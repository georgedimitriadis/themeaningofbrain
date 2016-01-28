# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 23:02:12 2014

@author: IntelligentSystem
"""

import subprocess

playerpath = r'C:\Users\IntelligentSystem\kampff.lab@gmail.com\software\utilities\video\Bonsai.VideoPlayer.Saving\Bonsai.VideoPlayer.exe'

def savemovie(filename,playbackrate,startframe,endframe,outfile):
    subprocess.call([playerpath,filename,str(playbackrate),str(startframe),str(endframe),outfile])