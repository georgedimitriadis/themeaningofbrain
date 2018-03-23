
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import BrainDataAnalysis.neuroseeker_specific_functions as nsf
from matplotlib.widgets import Button

data_folder = r'D:\Data\George\Neuroseeker Chronic Rat 22_1\2017_05_22\16_49_50' # first day
data_file = r'2017_05_22T16_49_50_Amp_S16_HP500Hz_LP3p5KHz_mV.bin'

data_folder = r'D:\Data\George\Neuroseeker Chronic Rat 22_1\2017_06_06\13_34_05' # last day
data_file = r'2017_06_06T13_34_05_Amp_S16_HP500Hz_LP3p5KHz_mV.bin'

data_folder = r'D:\Data\George\Neuroseeker Chronic Rat 22_1\2017_06_02\13_20_21'
data_file = r'InBehaviour_2017-06-02T13_20_21_Amp_S16_mV.bin'



full_data = nsf.load_binary_amplifier_data(join(data_folder, data_file))


lfp_channels = np.arange(9, 1440, 20)

bad_channels = np.concatenate((np.arange(720, 760), np.arange(1200, 1240), np.arange(1320, 1440)))

channels_to_remove = np.concatenate((lfp_channels, bad_channels, nsf.references)).astype(np.int)
channels_to_remove = np.sort(channels_to_remove)

good_channels = np.arange(nsf.number_of_channels_in_binary_file)
good_channels = np.delete(good_channels, channels_to_remove).astype(np.int)


brain_regions = {'Parietal_Cortex': 8000, 'Hypocampus_CA1': 6230, 'Hypocampus_DG': 5760, 'Thalamus_LPMR': 4450,
                 'Thalamus_Posterior': 3500, 'Thalamus_VPM': 1930, 'SubThalamic': 1050}


good_channels_height = []
for channel in good_channels:
    row = int(channel / 8) * 2
    if ((channel % 8) + 1) % 2 != 1:
        row += 1
    good_channels_height.append(row * nsf.electrode_pitch)

good_channels_height = np.array(good_channels_height)


good_channels_per_region = {'Parietal_Cortex': good_channels[
                                np.where(np.logical_and(good_channels_height <= brain_regions['Parietal_Cortex'],
                                                        good_channels_height > brain_regions['Hypocampus_CA1']))],
                            'Hypocampus_CA1': good_channels[
                                np.where(np.logical_and(good_channels_height <= brain_regions['Hypocampus_CA1'],
                                                        good_channels_height > brain_regions['Hypocampus_DG']))],
                            'Hypocampus_DG': good_channels[
                                np.where(np.logical_and(good_channels_height <= brain_regions['Hypocampus_DG'],
                                                        good_channels_height > brain_regions['Thalamus_LPMR']))],
                            'Thalamus_LPMR': good_channels[
                                np.where(np.logical_and(good_channels_height <= brain_regions['Thalamus_LPMR'],
                                                        good_channels_height > brain_regions['Thalamus_Posterior']))],
                            'Thalamus_Posterior': good_channels[
                                np.where(np.logical_and(good_channels_height <= brain_regions['Thalamus_Posterior'],
                                                        good_channels_height > brain_regions['Thalamus_VPM']))],
                            'Thalamus_VPM': good_channels[
                                np.where(np.logical_and(good_channels_height <= brain_regions['Thalamus_VPM'],
                                                        good_channels_height > brain_regions['SubThalamic']))],
                            'SubThalamic': good_channels[
                                np.where(good_channels_height <= brain_regions['SubThalamic'])]
                            }


shown_channels = good_channels_per_region['SubThalamic']

#plt.plot(nsf.spread_data(full_data[:, 2000000:2010000], channels_height=nsf.all_channels_height_on_probe, channels_used=shown_channels, row_spacing=1).T)


time_window = 0.5
time = 120
fig, ax = plt.subplots()
ax.set_autoscale_on(True)
plt.subplots_adjust(bottom=0.2)
tps = int(time * nsf.sampling_freq)
tpe = int((time + time_window) * nsf.sampling_freq)
time_axis = np.arange(time, (time + time_window), 1 / nsf.sampling_freq)
dat = nsf.spread_data(full_data[:, tps:tpe],
                      channels_height=nsf.all_channels_height_on_probe,
                      channels_used=shown_channels,
                      row_spacing=1)
lines = plt.plot(time_axis, dat.T)



class Index(object):
    time = time

    def next(self, event):
        self.time += time_window
        tps = int(self.time * nsf.sampling_freq)
        tpe = int((self.time + time_window) * nsf.sampling_freq)
        ydata = nsf.spread_data(full_data[:, tps:tpe],
                              channels_height=nsf.all_channels_height_on_probe,
                              channels_used=shown_channels,
                              row_spacing=1)
        time_axis = np.arange(self.time, (self.time + time_window), 1 / nsf.sampling_freq)
        for l in range(len(lines)):
            lines[l].set_data(time_axis, ydata[l, :])
        ax.relim()
        ax.autoscale(True, 'both', None)
        plt.draw()

    def prev(self, event):
        self.time -= time_window
        tps = int(self.time * nsf.sampling_freq)
        tpe = int((self.time + time_window) * nsf.sampling_freq)
        ydata = nsf.spread_data(full_data[:, tps:tpe],
                                channels_height=nsf.all_channels_height_on_probe,
                                channels_used=shown_channels,
                                row_spacing=1)
        time_axis = np.arange(self.time, (self.time + time_window), 1 / nsf.sampling_freq)
        for l in range(len(lines)):
            lines[l].set_data(time_axis, ydata[l, :])
        ax.relim()
        ax.autoscale(True, 'both', None)
        plt.draw()

callback = Index()
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)

plt.show()


# Get some noise values for the middle of the signal (3 million points = 2.5 minutes)
median = np.median(np.abs(full_data[good_channels,5000000:8000000]), axis = 1)
noise = median/0.6745
cortex_average_noise = np.mean(noise[917:])
cortex_std_noise = np.std(noise[917:])
hip_average_noise = np.mean(noise[705:917])
hip_std_noise = np.std(noise[705:917])
thal_average_noise = np.mean(noise[:705])
thal_std_noise = np.std(noise[:705])

