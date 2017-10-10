__author__ = 'Kampff-Lab-Analysis2'


import io_csv
import numpy as np

def load_colum_from_csv(filename, column, type='int'):
    data = []
    result = []
    with open(filename, 'rt', encoding='utf8') as csvfile:
        incsv = io_csv.reader(csvfile)
        for row in incsv:
            data.append(row)
        for row in data:
            if type == 'int':
                result.append(int(row[column]))
            elif type == 'datetime':
                result.append(np.datetime64(row[column]))
    return np.array(result)