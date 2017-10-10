
import numpy as np
import os.path as op
from ExperimentSpecificCode.Neuroseeker_Auditory_Double_2017_03_28.Stimulus import arnes_basic_analysis as ba

data_path = r'F:\Neuroseeker\Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes'

blocks = ba.load_data_neo(op.expanduser(data_path))

block = blocks['Angled']
seg = [s for s in block.segments if s.name == 'ToneSequence'][0]
params = seg.annotations['params']
event_info = seg.annotations['events']

T = np.unique([ev['Duration'] for ev in event_info])
levels = params['levels'][:3]  # throw away the 90db
frequencies = params['frequencies']

