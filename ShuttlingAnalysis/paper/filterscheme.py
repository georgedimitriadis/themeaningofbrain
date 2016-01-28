# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 05:19:40 2014

@author: Gonçalo
"""

import matplotlib.pyplot as plt

def durationheight(cr):
    plt.figure()
    plt.plot(cr.duration,cr.yhead_max,'.')
    plt.xlabel('duration (s)')
    plt.ylabel('max height (cm)')
    plt.title('all crossing trials')
    
def durationspeed25(cr):
    plt.figure()
    plt.plot(cr.duration,cr.xhead_speed_25,'.')
    plt.xlabel('duration (s)')
    plt.ylabel('speed Q1 (cm / s)')
    plt.title('all crossing trials')

durationheight(cr)
xmin,xmax = plt.xlim()
plt.hlines(0,xmin,xmax,'r')
plt.hlines(20.42,xmin,xmax,'r')

cut = cr.query('yhead_max > 0 and yhead_max < 20.42')
durationheight(cut)

durationspeed25(cut)

cut = cr.query('yhead_max > 0 and yhead_max < 20.42 and xhead_speed_25 == 0')
durationspeed25(cut)
durationheight(cut)