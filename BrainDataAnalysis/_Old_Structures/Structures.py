# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:49:57 2014

@author: George Dimitriadis
"""

from BrainDataAnalysis._Old_Structures.Constants import Dimension as dm
import numpy as np

class DataDimensions:
    def __init__(self, _dimOrder):
        self.order = _dimOrder

    def __str__(self):
        strToPrint = ""
        for count, dimension in enumerate(self.order):
            strToPrint = strToPrint + dimension
            if count<len(self.order)-1:
                strToPrint = strToPrint+", "
        return strToPrint
        
    @classmethod
    def fromIndividualDimensions(cls, *args):
        dimOrder = []
        for count, dimension in enumerate(args):
            if dimension is not dm.CHANNELS:
                if dimension is not dm.TIME:
                    if dimension is not dm.TRIALS:
                        if dimension is not dm.FREQUENCY:
                            raise TypeError("The arguments must be strings defined in Constants.Dimension")
            dimOrder.append(dimension)
        return cls(dimOrder)
    
    @classmethod
    def fromArrayOfDimensions(cls, dimOrder):
        for count, dimension in enumerate(dimOrder):
            if dimension is not dm.CHANNELS:
                if dimension is not dm.TIME:
                    if dimension is not dm.TRIALS:
                        if dimension is not dm.FREQUENCY:
                            raise TypeError("The arguments must be strings defined in Constants.Dimension")
        return cls(dimOrder)
    
    def getDimensionsAsString(self):
        strToPrint = ""
        for count, dimension in enumerate(self.order):
            strToPrint = strToPrint + dimension
            if count<len(self.order)-1:
                strToPrint = strToPrint+", "
        return strToPrint
    
    
    
class Data:
    def __init__(self, _dataMatrix, _dimOrder):
        self.dataMatrix = _dataMatrix
        if _dimOrder.__class__.__name__ is not 'DataDimensions':
            raise TypeError("Second argument must be a DataDimensions")
        self.order = _dimOrder
        
    @classmethod
    def withTimeAxis(cls, _dataMatrix, _dimOrder, _timeAxis):
        cls.dataMatrix = _dataMatrix
        if _dimOrder.__class__.__name__ is not 'DataDimensions':
            raise TypeError("Second argument must be a DataDimensions")
        cls.order = _dimOrder
        cls.timeAxis = _timeAxis
        return cls
    
    @classmethod
    def withFrequencyAxis(cls, _dataMatrix, _dimOrder, _freqAxis):
        cls.dataMatrix = _dataMatrix
        if _dimOrder.__class__.__name__ is not 'DataDimensions':
            raise TypeError("Second argument must be a DataDimensions")
        cls.order = _dimOrder
        cls.frequencyAxis = _freqAxis
        return cls
    
    @classmethod
    def withChannelNames(cls, _dataMatrix, _dimOrder, _channels):
        for count, channelName in enumerate(_channels):
            if type(channelName) is not str:
                raise TypeError("All elements in the argument list must be strings")
        cls.channels = _channels
        cls.dataMatrix = _dataMatrix
        if _dimOrder.__class__.__name__ is not 'DataDimensions':
            raise TypeError("Second argument must be a DataDimensions")
        cls.order = _dimOrder
        return cls
    
    @classmethod   
    def addTimeAxis(cls, _timeAxis):
        cls.timeAxis = timeAxis
        return cls
    
    @classmethod
    def addFrequencyAxis(cls, _freqAxis):
        cls.frequencyAxis = _freqAxis
        return cls
 
    @classmethod
    def addChannelNames(cls, _channels):
        for count, channelName in enumerate(_channels):
            if type(channelName) is not str:
                raise TypeError("All elements in the argument list must be strings")
        cls.channels = _channels
        return cls
    
    def getChannels(cls):
        return cls.channels
    
    def shape(cls):
        return np.shape(cls.dataMatrix)
        
    def dimensions(cls):
        return cls.order
        
    def getDimensionsAsString(cls):
        return cls.order.getDimensionsAsString()



class EventsData:
    def __init__(self, _eventValues, _samplePointsOfEvents, _eventName):
        self.samplePoints = _samplePointsOfEvents
        self.eventValues = _eventValues
    
    


class Trial:
    def __init__(self, _beginSample, _endSample, _offsetSample):
        if _beginSample>=_endSample:
            raise AttributeError("The begin sample has to be smaller than the end sample")
        self.beginSample = _beginSample
        self.endSample = _endSample
        self.offsetSample = _offsetSample
    
         
    def __setattr__(self, name, value):
        self.__dict__[name]=value
    
    
    def withRealTime(self, Fsampling):  
        offset = self.offsetSample*(1.0/Fsampling)
        self.beginTime = offset
        self.endTime = (self.endSample - self.beginSample)/Fsampling + offset
        self.timeAxis = np.arange(self.beginTime,self.endTime,1.0/Fsampling)
        return self
    
    
    def withIndex(self,_index):
        self.index = _index
        return self
    
    
    def withExtraInfo(self,**kwargs):
        for key, value in kwargs.iteritems():
            self.__setattr__(key,value)
        return self
    
    
    
class Trials:
    def __init__(self, beginSamples, endSamples, offsetSamples, Fsampling=None, **kwargs):
        """
        **kwargs must be a key value set of pairs where each key is a string and each value is an array as long as the xxxSamples arrays
        """
        self.allTrials=[]
        __bsLen  = len(beginSamples)
        __esLen = len(endSamples)
        __osLen = len(offsetSamples)
        if __bsLen!=__esLen or __esLen!=__osLen:
            raise AttributeError("The length of the begin, end and offset arrays must be the same")
        for key, value in kwargs.iteritems():
            if len(value) is not __bsLen:
                 raise AttributeError("The length of the array value in the %s key must be the same to the length of the xxxSamples arrays"%key)    
        numberOfTrials = __bsLen
        if Fsampling is not None:
            for i in range(numberOfTrials):
                self.allTrials.append( Trial(beginSamples[i], endSamples[i], offsetSamples[i]).withRealTime(Fsampling).withIndex(i))
        else:
           for i in range(numberOfTrials):
                self.allTrials.append(Trial(beginSamples[i], endSamples[i], offsetSamples[i]).withIndex(i))
        for k in range(numberOfTrials):     
            for key, value in kwargs.iteritems():
                a = {key:value[k]}
                self.allTrials[k].withExtraInfo(**a)