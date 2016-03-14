

import numpy as np
from Layouts.Probes import klustakwik_prb_generator as kpg
from Layouts.Probes import probes_imec as pi


r1 = np.array([103, 99, 95, 91, 87, 66, 82, 108, 47, 43, 1, 61, 57, 36, 34, 32,	30,	28,	26,	24,	22,	20])
r2 = np.array([104, 117, 121, 125, 71, 74, 78, 112, 51, 55, 62, 4, 6, 8, 10, 12, 14, 21, 19, 16])
r3 = np.array([102,	98, 94, 90, 86, 68, 85, 81, 46, 42, 38, 63, 59, 39, 37, 35, 33, 31, 29, 27, 25, 23])
r4 = np.array([109,	107, 105, 116, 118,	120, 122, 124, 126, 73,	69,	64,	75,
               77,	79,	113, 48, 50, 52, 54, 56, 0, 60, 3, 5, 7, 9, 11, 13, 15, 18])

bad_channels = np.concatenate((r1, r2, r3, r4))

all_electrodes, channel_positions = pi.create_128channels_imec_prb(bad_channels=bad_channels)

filename = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik_128to32_probe\128to32ch_passive_imec.prb'
kpg.generate_prb_file(filename, all_electrodes_array=all_electrodes, channel_number=128, steps_r=2, steps_c=2)