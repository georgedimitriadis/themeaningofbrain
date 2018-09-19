
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np
import sys
import matplotlib
matplotlib.use('Qt5Agg')
import configparser
from os.path import join, isfile, dirname, isdir
import pandas as pd

try:
    from . import custom_viewbox
except:  # Exception
    from GUIs.Tsne_spikesort import custom_viewbox as cv
try:
    from . import matplotlib_widget
except:  # Exception
    from GUIs.Tsne_spikesort import matplotlib_widget
try:
    from . import helper_functions as hf
except:  # Exception
    from GUIs.Tsne_spikesort import helper_functions as hf
try:
    from . import spike_heatmap as sh
except:  # Exception
    from GUIs.Tsne_spikesort import spike_heatmap as sh

global currently_selected_spikes
currently_selected_spikes = np.empty(0)

global currently_selected_templates
currently_selected_templates = np.empty(0)

global avg_electrode_curves
avg_electrode_curves = []

global data_of_selected_spikes
data_of_selected_spikes = np.empty(0)

global number_of_spikes
number_of_spikes = 0

global tsne_spots
tsne_spots = np.empty(0)

global spike_info
spike_info = pd.DataFrame()

global duration_of_time_plot
duration_of_time_plot = 0

global sampling_frequency
sampling_frequency = 0

global raw_data
raw_data = np.empty(0)

global channel_map
channel_map = []

global number_of_channels_in_binary_file
number_of_channels_in_binary_file = 0

global prb_file
prb_file = ''

global num_of_shanks_for_vis
num_of_shanks_for_vis = 1


def spikesort_gui(load_previous_dataset=True):

    def load_config():
        global raw_data
        global sampling_frequency
        global number_of_channels_in_binary_file
        global prb_file
        global duration_of_time_plot
        global spike_info
        global channel_map
        global num_of_shanks_for_vis

        config = configparser.ConfigParser()
        config.read('defaults.ini')

        if not 'SPIKES' in config:
            return

        all_exist = True

        spike_info_file = config['SPIKES']['spike_info_file']
        if 'spike_info.df' in spike_info_file:
            spike_info = pd.read_pickle(spike_info_file)

            update_tamplates_table()
            update_scater_plot()

            file_name = config['BINARY DATA FILE']['binary_data_filename']
            if not isfile(file_name):
                all_exist = False

            prb_file = config['PROBE']['prb_file']
            if not isfile(prb_file):
                all_exist = False

            channel_map_file = config['PROBE']['channel_map_file']
            if not isfile(channel_map_file):
                all_exist = False
            else:
                channel_map = np.squeeze(np.load(channel_map_file))

            if all_exist:
                type = config['BINARY DATA FILE']['type']
                types = {'int16': np.int16,
                         'uint16': np.uint16,
                         'int32': np.int32,
                         'uint32': np.uint32,
                         'int64': np.int64,
                         'uint32': np.uint64,
                         'float16': np.float16,
                         'float32': np.float32,
                         'float64': np.float64
                         }
                number_of_channels_in_binary_file = int(config['BINARY DATA FILE']['number_of_channels_in_binary_file'])
                order = config['BINARY DATA FILE']['order']

                raw_data = np.memmap(file_name, mode='r', dtype=types[type])
                raw_data = np.reshape(raw_data,
                                      (number_of_channels_in_binary_file,
                                       int(raw_data.shape[0] / number_of_channels_in_binary_file)),
                                      order=order)

                sampling_frequency = int(config['BINARY DATA FILE']['sampling_frequency'])

                electrodes = len(channel_map)
                for i in range(electrodes):
                    electrode_curve = pg.PlotCurveItem()
                    spikes_time_plot.addItem(electrode_curve)
                    avg_electrode_curves.append(electrode_curve)

                duration_of_time_plot = float(config['TIME PLOT']['duration_of_time_plot'])

                num_of_shanks_for_vis = config['PROBE']['num_of_shanks_for_vis']
                if num_of_shanks_for_vis != 'None':
                    num_of_shanks_for_vis = int(num_of_shanks_for_vis)
                else:
                    num_of_shanks_for_vis = None

            else:
                print('Some file paths in the configuration file do not point to existing files. '
                      'Load your data manually')

        elif spike_info_file == 'empty':
            print('Configuration file is empty. Load your data manually')

    def action_load_new_data():
        config = configparser.ConfigParser()
        config.read('defaults.ini')

        if config['SPIKES']['spike_info_file'] == 'empty':
            directory = None
        else:
            directory = dirname(config['SPIKES']['spike_info_file'])

        fname = QtWidgets.QFileDialog.getOpenFileName(caption='Load spike_info.df file',
                                                      directory=directory)
        spike_info_file = fname[0]
        if 'spike_info.df' in spike_info_file:
            config['SPIKES']['spike_info_file'] = spike_info_file
        else:
            print('Point to a file named spike_info.df')
            return

        fname = QtWidgets.QFileDialog.getOpenFileName(caption='Load binary data file',
                                                      directory=directory)
        binary_data_filename = fname[0]
        config['BINARY DATA FILE']['binary_data_filename'] = binary_data_filename

        fname = QtWidgets.QFileDialog.getOpenFileName(caption='Load prb file',
                                                      directory=directory)
        prb_file = fname[0]
        config['PROBE']['prb_file'] = prb_file

        fname = QtWidgets.QFileDialog.getOpenFileName(caption='Load channel map file',
                                                      directory=directory)
        channel_map_file = fname[0]
        config['PROBE']['channel_map_file'] = channel_map_file

        types = {'int16': np.int16,
                 'uint16': np.uint16,
                 'int32': np.int32,
                 'uint32': np.uint32,
                 'int64': np.int64,
                 'uint64': np.uint64,
                 'float16': np.float16,
                 'float32': np.float32,
                 'float64': np.float64
                 }
        type, ok = QtWidgets.QInputDialog.getItem(central_window, 'Binary Data Type',
                                                  'Input the type of the binary data file',
                                                  sorted(list(types.keys())))
        if ok:
            config['BINARY DATA FILE']['type'] = type

        order, ok = QtWidgets.QInputDialog.getItem(central_window, 'Binary Data Order',
                                                   'Input the order of the binary data file',
                                                   ['F', 'C'])
        if ok:
            config['BINARY DATA FILE']['order'] = order

        sampling_frequency, ok = QtWidgets.QInputDialog.getInt(central_window, 'Sampling Frequency',
                                                               'Input the sampling frequency of the binary data',
                                                               30000, 1000, 100000, 1)
        if ok:
            config['BINARY DATA FILE']['sampling_frequency'] = str(sampling_frequency)

        number_of_channels_in_binary_file, ok = QtWidgets.QInputDialog.getInt(central_window,
                                                                              'Number of Channels in Binary File',
                                                                              'Input the total number of channels in '
                                                                              'the binary data',
                                                                              128, 4, 10000, 1)
        if ok:
            config['BINARY DATA FILE']['number_of_channels_in_binary_file'] = str(number_of_channels_in_binary_file)

        duration_of_time_plot, ok = QtWidgets.QInputDialog.getInt(central_window, 'Duration of Time Plot',
                                                                  'Input the duration of the time plot in ms',
                                                                  4, 1, 20, 1)
        if ok:
            config['TIME PLOT']['duration_of_time_plot'] = str(duration_of_time_plot / 1000)

        num_of_shanks_for_vis, ok = QtWidgets.QInputDialog.getItem(central_window, 'Number of Shanks for Visual',
                                                                   'Input the number of pieces the probe should be '
                                                                   'visualized in (for long single shank probes). '
                                                                   'Use None for multi-shank probes with shanks '
                                                                   'defined in the prb file',
                                                                   ['None', '1', '2', '3', '4', '5', '6'])
        if ok:
            config['PROBE']['num_of_shanks_for_vis'] = num_of_shanks_for_vis

        with open('defaults.ini', 'w') as configfile:
            config.write(configfile)

        load_config()

    def action_save_info():
        global spike_info

        fname = QtWidgets.QFileDialog.getExistingDirectory(caption='Select directory to save the new '
                                                                   'spike_info.pd file. Any existing'
                                                                   ' spike_info.pd file will be overwritten.')
        print(fname)
        spike_info.to_pickle(join(fname, 'spike_info.df'))

        config = configparser.ConfigParser()
        config.read('defaults.ini')
        config.set('SPIKES', 'spike_info_file', join(fname, 'spike_info.df'))
        with open('defaults.ini', 'w') as configfile:
            config.write(configfile)

    def on_roi_selection(all_rois, freeform=False):
        global currently_selected_spikes
        global currently_selected_templates
        global tsne_spots
        global spike_info

        currently_selected_spikes = np.empty(0)
        points = np.array([tsne_spots[i]['pos'] for i in range(len(tsne_spots))])

        for i in range(len(all_rois)):
            if not freeform:
                shape = all_rois[i].parentBounds()
                bounds = shape
            else:
                shape = all_rois[i].shape()
                bounds = shape.boundingRect()

            selected = np.argwhere((points[:, 0] > bounds.x()) *
                                   (points[:, 0] < bounds.x() + bounds.width()) *
                                   (points[:, 1] > bounds.y()) *
                                   (points[:, 1] < bounds.y() + bounds.height())).flatten()
            if freeform:
                currently_selected_spikes = np.append(currently_selected_spikes,
                                                      [i for i in selected if
                                                shape.contains(QtCore.QPointF(points[i][0], points[i][1]))])
            else:
                currently_selected_spikes = np.append(currently_selected_spikes, selected)

        currently_selected_templates = hf.find_templates_of_spikes(spike_info, currently_selected_spikes)
        selected_templates_text_box.setText('Selected templates: ' + ', '.join(str(template) for template in
                                                                               currently_selected_templates))
        print('Number of spikes selected: ' + str(len(currently_selected_spikes)))

        show_selected_points()

    def on_roi_deletion():
        global currently_selected_spikes
        global currently_selected_templates

        currently_selected_spikes = np.empty(0)
        currently_selected_templates = np.empty(0)

    def show_selected_points():
        global spike_info

        scatter_selected_item.clear()
        scatter_selected_item.setData(pos=np.array([spike_info['tsne_x'].iloc[currently_selected_spikes].as_matrix(),
                                                    spike_info['tsne_y'].iloc[currently_selected_spikes].as_matrix()]).T)
        scatter_selected_item.setBrush((255, 255, 255))

    def update_scater_plot():
        global number_of_spikes
        global tsne_spots
        global spike_info

        scatter_item.clear()
        scatter_selected_item.clear()

        number_of_spikes = len(spike_info)

        progdialog = QtWidgets.QProgressDialog()
        progdialog.setGeometry(500, 500, 400, 40)
        progdialog.setWindowTitle('Loading t-sne plot')
        progdialog.setMinimum(0)
        progdialog.setMaximum(5)
        progdialog.setWindowModality(QtCore.Qt.WindowModal)

        progdialog.setLabelText('Generating colors _____________________________________')
        progdialog.show()
        progdialog.setValue(0)
        QtWidgets.QApplication.processEvents()

        color = [pg.intColor(spike_info['template_after_sorting'].iloc[i], hues=50, values=1, maxValue=255, minValue=150,
                             maxHue=360, minHue=0,
                             sat=255, alpha=255) for i in range(number_of_spikes)]

        progdialog.setLabelText('Generating brushes ____________________________________')
        progdialog.setValue(1)

        brush = [pg.mkBrush(color[i]) for i in range(number_of_spikes)]

        progdialog.setLabelText('Generating positions __________________________________')
        progdialog.setValue(2)

        tsne_spots = [{'pos': [spike_info['tsne_x'].iloc[i], spike_info['tsne_y'].iloc[i]],
                       'data': 1, 'brush': brush[i]} for i in range(number_of_spikes)]

        progdialog.setLabelText('Adding points to plot __________________________________')
        progdialog.setValue(3)

        scatter_item.addPoints(tsne_spots)

        progdialog.setLabelText('Generating symbols ... Almost done ______________________')
        progdialog.setValue(4)

        symbol = []
        for i in range(number_of_spikes):
            symbol.append(hf.symbol_from_type(spike_info['type_after_sorting'].iloc[i]))

        scatter_item.setSymbol(symbol)

        progdialog.setValue(5)
        progdialog.close()

    def update_average_spikes_plot():
        global currently_selected_spikes
        global data_of_selected_spikes
        global spike_info
        global channel_map

        electrodes = len(channel_map)

        half_duration = duration_of_time_plot / 2
        time_points = int(half_duration * sampling_frequency)
        time_axis = np.arange(-half_duration, half_duration, 1 / sampling_frequency)

        data_of_selected_spikes = np.zeros((electrodes, 2 * time_points))
        times = spike_info['times'].iloc[currently_selected_spikes]

        progdialog = QtWidgets.QProgressDialog()
        progdialog.setMinimum(0)
        progdialog.setMaximum(len(times))
        progdialog.setWindowTitle('Loading spikes')
        progdialog.setGeometry(500, 500, 400, 40)
        progdialog.setWindowModality(QtCore.Qt.WindowModal)

        progdialog.show()

        prog = 0
        for t in times:
            add_data = raw_data[channel_map, int(t-time_points):int(t+time_points)]
            data_of_selected_spikes = data_of_selected_spikes + add_data
            prog += 1
            progdialog.setValue(prog)
            progdialog.setLabelText('Loading spike: ' + str(prog) + ' of ' + str(len(times)))
            QtWidgets.QApplication.processEvents()

        data_of_selected_spikes = data_of_selected_spikes / len(times)

        for i in np.arange(electrodes):
            avg_electrode_curves[i].setData(time_axis, data_of_selected_spikes[i, :])
            avg_electrode_curves[i].setPen(pg.mkPen(color=(i, number_of_channels_in_binary_file * 1.3),
                                                    width=0.5))

        progdialog.setValue(len(times))
        progdialog.close()

    def update_tamplates_table():
        global number_of_spikes
        global spike_info

        templates_table.setData(hf.get_templates_with_number_of_spikes_from_spike_info(spike_info))
        templates_table.setHorizontalHeaderLabels(['Template number', 'Number of spikes in template'])
        templates_table.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        templates_table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        header = templates_table.horizontalHeader()
        header.setResizeMode(0, QtGui.QHeaderView.ResizeToContents)
        header.setResizeMode(1, QtGui.QHeaderView.ResizeToContents)

    def update_heatmap_plot():
        global data_of_selected_spikes
        global channel_map
        global number_of_channels_in_binary_file
        global num_of_shanks_for_vis

        connected = channel_map
        connected_binary = np.in1d(np.arange(number_of_channels_in_binary_file), connected)
        bad_channels = np.squeeze(np.argwhere(connected_binary==False).astype(np.int))

        sh.create_heatmap_on_matplotlib_widget(heatmap_plot, data_of_selected_spikes, prb_file, window_size=60,
                                               bad_channels=bad_channels, num_of_shanks=num_of_shanks_for_vis,
                                               rotate_90=True, flip_ud=False,flip_lr=False)
        heatmap_plot.draw()

    def update_autocorelogram():
        global currently_selected_spikes
        global spike_info

        spike_times = spike_info['times'].iloc[currently_selected_spikes].as_matrix().astype(np.int64)
        diffs, norm = hf.crosscorrelate_spike_trains(spike_times, spike_times, lag=1000)
        hist, edges = np.histogram(diffs, bins=50)
        autocorelogram_curve.setData(x=edges, y=hist, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))

    # On press button functions ---------
    def on_press_button_update():
        update_average_spikes_plot()
        update_autocorelogram()
        update_heatmap_plot()

    def on_tsne_color_scheme_combo_box_change(index):
        global number_of_spikes
        global spike_info

        progdialog = QtWidgets.QProgressDialog()
        progdialog.setGeometry(500, 500, 400, 40)
        progdialog.setWindowTitle('Changing spike color scheme')
        progdialog.setMinimum(0)
        progdialog.setMaximum(2)
        progdialog.setWindowModality(QtCore.Qt.WindowModal)

        progdialog.setLabelText('Generating colors and symbols ...                  ')
        progdialog.show()
        progdialog.setValue(0)
        QtWidgets.QApplication.processEvents()

        spikes_to_change = range(number_of_spikes)

        if index == 0:
            color_scheme = spike_info['template_after_sorting']
            brush = [pg.intColor(color_scheme.iloc[i], hues=50, values=1, maxValue=255, minValue=150,
                                 maxHue=360, minHue=0, sat=255, alpha=255)
                     for i in spikes_to_change]
            symbol = []
            for i in spikes_to_change:
                symbol.append(hf.symbol_from_type(spike_info['type_after_sorting'].iloc[i]))
        elif index == 1:
            color_scheme = spike_info['template_after_cleaning']
            brush = [pg.intColor(color_scheme.iloc[i], hues=50, values=1, maxValue=255, minValue=150,
                                 maxHue=360, minHue=0, sat=255, alpha=255)
                     for i in spikes_to_change]
            symbol = []
            for i in spikes_to_change:
                symbol.append(hf.symbol_from_type(spike_info['type_after_cleaning'].iloc[i]))

        progdialog.setValue(1)
        progdialog.setLabelText('Applying colors and symbols ...                   ')
        QtWidgets.QApplication.processEvents()

        scatter_item.setBrush(brush)
        scatter_item.setSymbol(symbol)

        progdialog.setValue(2)
        progdialog.close()

    def on_templates_table_selection():
        global currently_selected_templates
        global currently_selected_spikes
        global spike_info

        currently_selected_templates = []
        rows_selected = templates_table.selectedItems()
        for i in np.arange(0, len(rows_selected), 2):
            currently_selected_templates.append(int(rows_selected[i].text()))

        currently_selected_templates = np.array(currently_selected_templates)

        currently_selected_spikes = np.empty(0)
        for selected_template in currently_selected_templates:
            currently_selected_spikes = np.append(currently_selected_spikes,
                                                  spike_info.loc[spike_info['template_after_sorting'] ==
                                                                 selected_template].index)

        show_selected_points()

    def on_press_button_show_cross_correlograms():
        global currently_selected_templates
        global spike_info

        num_of_templates = len(currently_selected_templates)
        if num_of_templates > 1:
            cross_cor_window.show()
            num_of_windows = (num_of_templates**2 - num_of_templates) / 2 + num_of_templates
            cross_corelogram_plots = []
            column = 0
            row = 0
            width = 1
            height = 1

            progdialog = QtWidgets.QProgressDialog()
            progdialog.setMinimum(0)
            progdialog.setMaximum(num_of_windows)
            progdialog.setWindowTitle('Making cross correlograms')
            progdialog.setGeometry(500, 500, 400, 40)
            progdialog.setWindowModality(QtCore.Qt.WindowModal)

            progdialog.show()
            prog = 0

            for window_index in np.arange(num_of_windows):
                cross_corelogram_plots.append(pg.PlotWidget())
                cross_cor_grid_layout.addWidget(cross_corelogram_plots[-1], column, row, height, width)

                spike_times_a = spike_info['times'].loc[
                    spike_info['template_after_sorting'] == currently_selected_templates[column]].as_matrix().astype(np.int64)
                spike_times_b = spike_info['times'].loc[
                    spike_info['template_after_sorting'] == currently_selected_templates[row]].as_matrix().astype(np.int64)

                diffs, norm = hf.crosscorrelate_spike_trains(spike_times_a, spike_times_b, lag=3000)

                cross_corelogram_curve = pg.PlotCurveItem()
                cross_corelogram_plots[-1].addItem(cross_corelogram_curve)

                hist, edges = np.histogram(diffs, bins=100)
                cross_corelogram_curve.setData(x=edges, y=hist, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))

                row += width
                if row > column:
                    column += height
                    row = 0

                prog += 1
                progdialog.setValue(prog)
                progdialog.setLabelText('Making cross correlogram: ' + str(prog) + ' of ' + str(num_of_windows))
                QtWidgets.QApplication.processEvents()

            progdialog.setValue(num_of_windows)
            progdialog.close()

    def on_press_button_remove_from_template():
        global currently_selected_spikes
        global spike_info

        if len(currently_selected_spikes) > 0:
            spike_info.loc[currently_selected_spikes, 'template_after_sorting'] = 0
            update_tamplates_table()

    def on_press_button_make_template():
        global currently_selected_spikes
        global spike_info

        if len(currently_selected_spikes) > 0:
            all_templates = spike_info['template_after_sorting'].as_matrix()
            max_template = np.max(all_templates)
            index, ok = QtWidgets.QInputDialog.getInt(central_window, 'New Template Index',
                                                     'Input the index of the new template',
                                                     max_template + 1, max_template + 1, max_template + 10000, 1)
            if ok:
                spike_info.loc[currently_selected_spikes, 'template_after_sorting'] = index

            item, ok = QtWidgets.QInputDialog.getItem(central_window, 'New Template Type',
                                                      'Input the type of the new template',
                                                      ['Noise', 'Single Unit', 'Single Unit Contaminated',
                                                       'Single Unit Putative', 'MUA', 'Unspecified 1',
                                                       'Unspecified 2', 'Unspecified 3'])
            if ok:
                print(item)
            update_tamplates_table()
    # ----------------------------------

    # Main Window ----------------------
    app = QtGui.QApplication([])
    main_window = QtGui.QMainWindow()

    main_toolbar = main_window.addToolBar('Tools')
    main_window.setWindowTitle('Spikesort on T-sne')
    main_window.resize(1300, 900)
    central_window = pg.GraphicsLayoutWidget()
    main_window.setCentralWidget(central_window)
    grid_layout = QtGui.QGridLayout()
    central_window.setLayout(grid_layout)
    # ----------------------------------

    # T-sne Scatter Plot ---------------
    custom_viewbox = cv.CustomViewBox()
    scatter_plot = pg.PlotWidget(title='T-sne plot', viewBox=custom_viewbox)
    grid_layout.addWidget(scatter_plot, 0, 0, 2, 4)

    scatter_plot.plotItem.ctrlMenu = None
    scatter_plot.scene().contextMenu = None

    scatter_item = pg.ScatterPlotItem(size=7, pen=None, pxMode=True)

    scatter_plot.addItem(scatter_item)

    scatter_selected_item = pg.ScatterPlotItem(size=8, pen=None, pxMode=True)
    scatter_plot.addItem(scatter_selected_item)
    # ---------------------------------

    # Average spike time course plot --
    spikes_time_plot = pg.PlotWidget(title='Average time plot of selected spikes')
    grid_layout.addWidget(spikes_time_plot, 2, 0, 2, 2)
    # ---------------------------------

    # Autocorelogram plot -------------
    autocorelogram_plot = pg.PlotWidget(title='Autocorrelogram')
    grid_layout.addWidget(autocorelogram_plot, 2, 2, 2, 2)
    autocorelogram_curve = pg.PlotCurveItem()
    autocorelogram_plot.addItem(autocorelogram_curve)
    # ---------------------------------

    # Heatmap plot --------------------
    heatmap_plot = matplotlib_widget.MatplotlibWidget(toolbar_on=False)
    grid_layout.addWidget(heatmap_plot, 0, 4, 6, 2)
    # ---------------------------------

    # Templates Table widget -----------
    templates_table = pg.TableWidget(editable=False, sortable=True)
    templates_table.cellClicked.connect(on_templates_table_selection)
    grid_layout.addWidget(templates_table, 0, 6, 3, 1)
    # ----------------------------------

    # Cross correlogram window ---------
    cross_cor_window = QtGui.QMainWindow()

    cross_cor_window.setWindowTitle('Cross correlograms')
    cross_cor_window.resize(1300, 900)
    central_cross_cor_window = pg.GraphicsLayoutWidget()
    cross_cor_window.setCentralWidget(central_cross_cor_window)
    cross_cor_grid_layout = QtGui.QGridLayout()
    central_cross_cor_window.setLayout(cross_cor_grid_layout)
    # ----------------------------------

    # Tool bar -------------------------
    scroll_icon_file = join(sys.path[0], 'Icons',  'scroll_icon.png')
    zoom_icon_file = join(sys.path[0], 'Icons', 'zoom_icon.png')
    select_icon_file = join(sys.path[0], 'Icons',  'select_icon.png')
    select_freeform_icon_file = join(sys.path[0], 'Icons',  'select_freeform_icon.png')

    menu_bar = main_window.menuBar()
    file_menu = menu_bar.addMenu('File')
    load_data = QtWidgets.QAction('Load Data', main_window)
    load_data.triggered.connect(action_load_new_data)
    file_menu.addAction(load_data)
    save_info = QtWidgets.QAction('Save spike info', main_window)
    save_info.triggered.connect(action_save_info)
    file_menu.addAction(save_info)
    scroll = QtWidgets.QAction(QtGui.QIcon(scroll_icon_file), "scroll", main_window)
    scroll.triggered.connect(custom_viewbox.set_to_pan_mode)
    main_toolbar.addAction(scroll)
    zoom = QtWidgets.QAction(QtGui.QIcon(zoom_icon_file), "zoom", main_window)
    zoom.triggered.connect(custom_viewbox.set_to_rect_mode)
    main_toolbar.addAction(zoom)
    select = QtWidgets.QAction(QtGui.QIcon(select_icon_file), "select", main_window)
    select.triggered.connect(custom_viewbox.draw_square_roi)
    main_toolbar.addAction(select)
    select_freeform = QtWidgets.QAction(QtGui.QIcon(select_freeform_icon_file), 'select freeform', main_window)
    select_freeform.triggered.connect(custom_viewbox.draw_freeform_roi)
    custom_viewbox.connect_on_roi_select = on_roi_selection
    custom_viewbox.connect_on_roi_delete = on_roi_deletion
    main_toolbar.addAction(select_freeform)
    # ----------------------------------

    # Buttons --------------------------
    button_update = QtGui.QPushButton('Update')
    button_update.setStyleSheet("font-size:20px; font-family: Helvetica")
    button_update.clicked.connect(on_press_button_update)
    grid_layout.addWidget(button_update, 7, 0, 1, 1)

    button_show_cross_correlograms = QtGui.QPushButton('Show Crosscorrelograms')
    button_show_cross_correlograms.setStyleSheet("font-size:20px; font-family: Helvetica")
    button_show_cross_correlograms.clicked.connect(on_press_button_show_cross_correlograms)
    grid_layout.addWidget(button_show_cross_correlograms, 7, 1, 1, 1)

    tsne_color_scheme_combo_box = QtGui.QComboBox()
    tsne_color_scheme_combo_box.setStyleSheet("font-size:20px; font-family: Helvetica")
    tsne_color_scheme_combo_box.addItem('After sorting')
    tsne_color_scheme_combo_box.addItem('Before sorting')
    tsne_color_scheme_combo_box.activated.connect(on_tsne_color_scheme_combo_box_change)
    grid_layout.addWidget(tsne_color_scheme_combo_box, 7, 2, 1, 1)

    button_remove_from_template = QtGui.QPushButton('Remove selected spikes from template')
    button_remove_from_template.setStyleSheet("font-size:20px; font-family: Helvetica")
    button_remove_from_template.clicked.connect(on_press_button_remove_from_template)
    grid_layout.addWidget(button_remove_from_template, 4, 6, 1, 1)

    button_make_template = QtGui.QPushButton('Make new template from spikes')
    button_make_template.setStyleSheet("font-size:20px; font-family: Helvetica")
    button_make_template.clicked.connect(on_press_button_make_template)
    grid_layout.addWidget(button_make_template, 5, 6, 1, 1)
    # ----------------------------------

    # Text Box -------------------------
    selected_templates_text_box = QtWidgets.QLineEdit()
    selected_templates_text_box.setReadOnly(True)
    selected_templates_text_box.setWindowTitle('Selected Templates')
    selected_templates_text_box.setStyleSheet("font-size:20px; font-family: Helvetica")
    grid_layout.addWidget(selected_templates_text_box, 7, 3, 1, 2)
    # ----------------------------------

    if load_previous_dataset:
        load_config()

    main_window.show()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit(app.exec_())


spikesort_gui(load_previous_dataset=True)

