

import csv
from os import path
import pandas as pd
import numpy as np
import pandas as pd


data_folder = r'D:\\AK_33.1\2018_04_27-09_44\Data'
events_filder = path.join(data_folder, 'events')
events_file_name = path.join(data_folder, 'Events.csv')

pd_csv = pd.read_csv(events_file_name, delimiter=' | ( | )', engine='python', header=None, usecols=[0, 2, 4, 6])


col4 = pd_csv[4].str.replace(r'\(', '')
col6 = pd_csv[6].str.replace(r'\)', '')
for i in np.arange(len(col4)):
    pd_csv[4].iloc[i] = pd_csv[4].iloc[i].str.replace('*', col4[i], col6[i])