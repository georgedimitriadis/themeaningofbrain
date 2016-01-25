# -*- coding: utf-8 -*-
"""
Created on Tue Oct 07 13:17:13 2014

@author: Gonçalo
"""

import activitytables
import activityplots

subject = r'E:/Protocols/Shuttling/LightDarkServoStable/Data/JPAK_25'
stablerange = range(4,5)
randomrange = range(13,17)
act1 = activitytables.read_subjects(subject, days=stablerange)
act2 = activitytables.read_subjects(subject, days=randomrange)
cross1 = activitytables.read_subjects(subject, days=stablerange, selector=activitytables.crossings)
cross2 = activitytables.read_subjects(subject, days=randomrange, selector=activitytables.crossings)

validc1 = cross1[(cross1.label == 'valid') & (cross1.stepstate3)]
validc2 = cross2[(cross2.label == 'valid') & (cross2.stepstate3)]
traj1 = [act1[slice(s.start,s.stop)] for s in validc1.slices]
traj2 = [act2[slice(s.start,s.stop)] for s in validc2.slices]