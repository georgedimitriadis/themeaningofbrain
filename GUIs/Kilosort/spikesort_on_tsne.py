


import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from tsne_for_spikesort import io_with_cpp as io
from os.path import join
from BrainDataAnalysis import ploting_functions as pf


base_folder = r'D:\Data\Brain\Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Analysis\Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort'
exe_folder = r'E:\Projects\Analysis\Brain\spikesorting_tsne_bhpart\Barnes_Hut\x64\Release'
tsne = io.load_tsne_result(exe_folder)
cluster_info = np.load(join(base_folder, 'cluster_info.pkl'))
labels_dict = pf.generate_labels_dict_from_cluster_info_dataframe(cluster_info=cluster_info)





app = QtGui.QApplication([])
main_window = pg.GraphicsWindow(size=(1000,800), border=True)
main_window.setWindowTitle('Spikesort on T-sne')
main_window.resize(1300, 900)
layout = main_window.addLayout()





import sys
if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    sys.exit(app.exec_())