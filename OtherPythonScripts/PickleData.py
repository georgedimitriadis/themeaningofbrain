# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:42:05 2014

@author: IntelligentSystem
"""

import cPickle as pickle

def saveData(data, filename):
    if not ".p" in filename:
        filename = filename+".p"
    pickledFile = open(filename, "wb")
    pickle.dump( data, pickledFile,-1 )
    pickledFile.close()
    
    
def loadData(filename):
    if not ".p" in filename:
        filename = filename+".p"
    pickledFile = open(filename, "rb")
    data = pickle.load(pickledFile)
    pickledFile.close()
    return data