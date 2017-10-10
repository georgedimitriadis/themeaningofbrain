

import numpy as np



# Noise level for all channels
def SD_calculation(triggerdata):

    NumSites=32
    SD=np.zeros(NumSites)

    for i in range(NumSites):
        SD[i] = np.median(abs(triggerdata[i]))/0.6745

    return SD
