# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:15:37 2014

@author: George Dimitriadis
"""


class Dimension:
    CHANNELS = "channels"
    TIME = "time"
    TRIALS = "trials"
    FREQUENCY = "frequency"
    
    
class FilterDirection:
    ONE_PASS = "onepass"
    ONE_PASS_REVERSE = "onepass-reverse"
    TWO_PASS = "twopass"