# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 20:39:35 2014

@author: IntelligentSystem
"""

import cv2
import video
import imgproc
import numpy as np
import matplotlib.pyplot as plt

step3crop = (slice(496,615),slice(500,600))

def clustersteps(frames,K):
    data = np.float32(frames)
    termcrit = (cv2.TERM_CRITERIA_EPS, 30,0.1)
    return cv2.kmeans(data,K,termcrit,10,0)

def showcluster(frames,clusters,K):
    for k in range(K):
        cluster = [frames[i] for i in range(len(frames)) if clusters[i,0] == k]
        tiles = imgproc.tile(cluster,8,12)
        cv2.imshow('cluster{0}'.format(k),tiles[0])
        
def trycluster(data,vis,K):
    cv2.destroyAllWindows()
    ret,lbl,cent = clustersteps(data,K)
    showcluster(vis,lbl,K)