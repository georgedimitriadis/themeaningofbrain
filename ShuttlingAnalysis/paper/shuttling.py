# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:32:27 2013

@author: gonca_000
"""

import os
import poke

class shuttling:
    def __init__(self, path, leftpoke, rightpoke, rewards):
        self.path = path
        self.leftpoke = leftpoke
        self.rightpoke = rightpoke
        self.rewards = rewards
        
def genfromtxt(path):
    leftpokepath = os.path.join(path, 'left_poke.csv')
    leftrewardpath = os.path.join(path, 'left_rewards.csv')
    leftpoke = poke.genfromtxt(leftpokepath, leftrewardpath)
    
    rightpokepath = os.path.join(path, 'right_poke.csv')
    rightrewardpath = os.path.join(path, 'right_rewards.csv')
    rightpoke = poke.genfromtxt(rightpokepath, rightrewardpath)
    
    return shuttling(path, leftpoke, rightpoke, (leftpoke+rightpoke).rewards)