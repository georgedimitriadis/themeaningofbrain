# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:50:34 2013

@author: gonca_000
"""

import cv2
import bisect
import numpy as np

class video:        
    def __init__(self, videopath, timepath=None):
        self.path = videopath
        self.capture = cv2.VideoCapture(videopath)
        if timepath is not None:
            self.timestamps = np.genfromtxt(timepath,dtype=str)
        else:
            self.timestamps = None
        
    def __del__(self):
        del self.capture
        
    def frameindex(self, timestr):
        if self.timestamps is None:
            raise ValueError("video does not have timestamps")
        return bisect.bisect_left(self.timestamps,timestr)
        
    def frame(self, frameindex):
        self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,frameindex)
        result,frame = self.capture.read()
        return frame
        
    def movie(self, framestart, frameend):
        self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,framestart)
        while self.capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) < frameend:
            result,frame = self.capture.read()
            if result:
                yield frame
            else:
                break
            
def readframe(movie):
    index = movie.capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    result, frame = movie.capture.read()
    cv2.putText(frame,str(int(index)),(0,30),
                cv2.cv.CV_FONT_HERSHEY_COMPLEX,1,
                (255,255,255,255))
    return frame, index
            
def showmovie(movie,framestart=0,fps=0,frameend=None):
    key = 0
    interval = 0 if fps == 0 else int(1000.0 / fps)
    movie.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,framestart)
    frame, index = readframe(movie)
    while key != 27:
        cv2.imshow('win',frame)
        key = cv2.waitKey(interval)
        if key == 32: #space
            if interval > 0:
                interval = 0
            else:
                interval = 0 if fps == 0 else int(1000.0 / fps)
        if key == 2555904 or key < 0: #right arrow
            frame, index = readframe(movie)
            if index == frameend:
                interval = 0
            continue
        elif key == 2424832: #left arrow
            movie.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,index-1)
            frame, index = readframe(movie)
        elif key == 2228224: #page down
            movie.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,index+10)
            frame, index = readframe(movie)
        elif key == 2162688: #page up
            movie.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,index-10)
            frame, index = readframe(movie)
    cv2.destroyWindow('win')
            
def mvshow(winname, frames, interval = 1000.0 / 25, stepkey = -1):
    for frame in frames:
        key = None
        cv2.imshow(winname,frame)
        while key != stepkey:
            key = cv2.waitKey(interval)
        if key == 27: # Escape
            break
    cv2.destroyWindow(winname)