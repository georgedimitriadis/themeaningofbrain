__author__ = 'George Dimitriadis'


from PyQt5 import QtCore, QtWidgets
from ui_tfr_viewer import Ui_TFR_Viewer
import sys
import mne
import matplotlib.pyplot as plt


class TFR_Viewer():
    def __init__(self, power, baseline=None, mode='mean', tmin=None, tmax=None, fmin=None, fmax=None, vmin=None, vmax=None,
                 cmap=None, dB=False, colorbar=True,  x_label=None, y_label=None, picker=True,):

        app = QtWidgets.QApplication(sys.argv)
        window = QtWidgets.QMainWindow()
        self.ui = Ui_TFR_Viewer()
        self.ui.setupUi(window)
        self.connect_slots()

        if cmap is None:
            self.cmap = plt.cm.jet
        else:
            self.cmap = cmap

        self.tmin = tmin
        self.tmax = tmax
        self.colorbar = colorbar
        self.x_label = x_label
        self.y_label = y_label
        self.picker = picker

        times, self.freqs = power.times.copy(), power.freqs.copy()
        self.data = power.data
        self.tmin, self.tmax = times[0], times[-1]

        self.data, self.times, self.freqs, self.vmin, self.vmax = \
                mne.time_frequency.tfr._preproc_tfr(self.data, times, self.freqs, tmin, tmax,
                                                    fmin, fmax, mode, baseline, vmin, vmax, dB)
        self.ui.sb_channel.setMaximum(len(power.ch_names)-1)
        self.on_sb_channel_valueChanged(0)
        window.show()
        app.exec_()




    @QtCore.pyqtSlot(int)
    def on_sb_channel_valueChanged(self, chan):

        self.ui.mplg_tfr_viewer.all_sp_axes[0].clear()

        extent = (self.tmin, self.tmax, self.freqs[0], self.freqs[-1])
        self.ui.mplg_tfr_viewer.all_sp_axes[0].imshow(self.data[chan, :, :], extent=extent, aspect="auto", origin="lower",
                                                      vmin=self.vmin, vmax=self.vmax, picker=self.picker, cmap=self.cmap)

        if self.x_label is not None:
            plt.xlabel(self.x_label)
        if self.y_label is not None:
            plt.ylabel(self.y_label)
        #if self.colorbar:
        #    plt.colorbar()

        self.ui.mplg_tfr_viewer.canvas.draw()



    def connect_slots(self):
        self.ui.sb_channel.valueChanged.connect(self.on_sb_channel_valueChanged)




