
from __future__ import print_function, absolute_import, division
import numpy as np
from os.path import join, exists
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.widgets import MatplotlibWidget as ptl_widget
from GUIs.Kilosort import spike_heatmap as sh
from joblib import Parallel, delayed


def cleanup_kilosorted_data(base_folder, number_of_channels_in_binary_file, binary_data_filename,
                            probe_info_folder, probe_connected_channels_file, sampling_frequency=20000):


    channel_map = np.load(join(base_folder, 'channel_map.npy'))
    active_channel_map = np.squeeze(channel_map, axis=1)

    spike_templates = np.load(join(base_folder, r'spike_templates.npy'))
    template_feature_ind = np.load(join(base_folder, 'template_feature_ind.npy'))
    number_of_templates = template_feature_ind.shape[0]

    spike_times = np.load(join(base_folder, 'spike_times.npy')).astype(np.int)

    templates = np.load(join(base_folder, 'templates.npy'))

    num_of_channels = active_channel_map.size

    global current_template_index
    current_template_index = 0
    visibility_threshold = 2

    if exists(join(base_folder, 'template_marking.npy')):
        template_marking = np.load(join(base_folder, 'template_marking.npy'))
    else:
        template_marking = np.zeros(number_of_templates)
        np.save(join(base_folder, 'template_marking.npy'), template_marking)

    assert exists(join(base_folder, 'avg_spike_template.npy'))
    data = np.load(join(base_folder, 'avg_spike_template.npy'))

    time_points = data.shape[2]

    def get_visible_channels(current_template_index, visibility_threshold):
        median = np.median(np.nanmin(templates[current_template_index, :, :], axis=0))
        std = np.std(np.nanmin(templates[current_template_index, :, :], axis=0))
        points_under_median = np.argwhere(templates[current_template_index, :, :] < (median - visibility_threshold * std))
        channels_over_threshold = np.unique(points_under_median[:, 1])
        return channels_over_threshold

    def update_all_plots():
        update_average_spikes_plot()
        update_heatmap_plot()
        update_autocorelogram()
        update_marking_led()

    def update_average_spikes_plot():
        global current_template_index
        visible_channels = get_visible_channels(current_template_index=current_template_index,
                                                visibility_threshold=visibility_threshold)
        time_points = data.shape[2]
        total_time = time_points / sampling_frequency
        time_axis = np.arange(-(total_time/2), total_time/2, 1 / sampling_frequency)
        for i in np.arange(electrodes):
            electrode_curves[i].setData(time_axis, data[current_template_index, i, :])
            if i in visible_channels:
                electrode_curves[i].setPen(pg.mkPen((i, len(visible_channels) * 1.3)))
            else:
                electrode_curves[i].setPen(pg.mkPen(None))

    def get_all_spikes_form_template(template):
        visible_channels = get_visible_channels(current_template_index=current_template_index,
                                                visibility_threshold=visibility_threshold)
        num_of_channels = len(visible_channels)
        data_raw = np.memmap(binary_data_filename, dtype=np.int16, mode='r')

        number_of_timepoints_in_raw = int(data_raw.shape[0] / number_of_channels_in_binary_file)
        data_raw_matrix = np.reshape(data_raw, (number_of_channels_in_binary_file, number_of_timepoints_in_raw),
                                     order='F')

        spike_indices_in_template = np.argwhere(np.in1d(spike_templates, template))
        spike_times_in_template = np.squeeze(spike_times[spike_indices_in_template])

        too_early_spikes = np.squeeze(np.argwhere(spike_times_in_template < (time_points / 2)), axis=1)
        too_late_spikes = np.squeeze(
            np.argwhere(spike_times_in_template > number_of_timepoints_in_raw - (time_points / 2)), axis=1)
        out_of_time_spikes = np.concatenate((too_early_spikes, too_late_spikes))
        spike_indices_in_template = np.delete(spike_indices_in_template, out_of_time_spikes)
        num_of_spikes_in_template = spike_indices_in_template.shape[0]

        data = np.zeros((num_of_spikes_in_template, num_of_channels, time_points))

        for spike_in_template in spike_indices_in_template:
            data[spike_in_template, :, :] = data_raw_matrix[visible_channels,
                                                            spike_times[spike_in_template] - (time_points / 2):
                                                            spike_times[spike_in_template] + (time_points / 2)]

        return data



    '''
    def update_heatmap_plot():
        prb_file = join(probe_info_folder, 'prb.txt')
        connected = np.squeeze(np.load(join(probe_info_folder, probe_connected_channels_file)))
        bad_channels = np.squeeze(np.argwhere(connected == False).astype(np.int))
        image, _ = sh.create_heatmap(data[current_template_index], prb_file, window_size=60,
                                     bad_channels=bad_channels, num_of_shanks=5, rotate_90=True, flip_ud=False,
                                     flip_lr=False)
        heatmap_plot.setImage(image)
    '''

    def update_heatmap_plot():
        prb_file = join(probe_info_folder, 'prb.txt')
        connected = np.squeeze(np.load(join(probe_info_folder, probe_connected_channels_file)))
        bad_channels = np.squeeze(np.argwhere(connected == False).astype(np.int))
        sh.create_heatmap_on_matplotlib_widget(heatmap_plot, data[current_template_index], prb_file, window_size=60,
                                               bad_channels=bad_channels, num_of_shanks=5, rotate_90=True, flip_ud=False,
                                               flip_lr=False)
        heatmap_plot.draw()


    def crosscorrelate_spike_trains(spike_times_train_1, spike_times_train_2, lag=None):
        if spike_times_train_1.size < spike_times_train_2.size:
            if lag is None:
                lag = np.ceil(10 * np.mean(np.diff(spike_times_train_1)))
            reverse = False
        else:
            if lag is None:
                lag = np.ceil(20 * np.mean(np.diff(spike_times_train_2)))
            spike_times_train_1, spike_times_train_2 = spike_times_train_2, spike_times_train_1
            reverse = True

        # calculate cross differences in spike times
        differences = np.array([])
        for k in np.arange(0, spike_times_train_1.size):
            differences = np.append(differences, spike_times_train_1[k] - spike_times_train_2[np.nonzero(
                (spike_times_train_2 > spike_times_train_1[k] - lag)
                 & (spike_times_train_2 < spike_times_train_1[k] + lag)
                 & (spike_times_train_2 != spike_times_train_1[k]))])
        if reverse is True:
            differences = -differences
        norm = np.sqrt(spike_times_train_1.size * spike_times_train_2.size)
        return differences, norm


    def update_autocorelogram():
        global current_template_index
        spike_indices_in_template = np.argwhere(np.in1d(spike_templates, current_template_index))
        diffs, norm = crosscorrelate_spike_trains(spike_times[spike_indices_in_template].astype(np.int64),
                                                  spike_times[spike_indices_in_template].astype(np.int64),
                                                  lag=3000)
        hist, edges = np.histogram(diffs, bins=100)
        autocorelogram_curve.setData(x=edges, y=hist, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))

        number_of_spikes = len(spike_indices_in_template)
        plot_average_spikes_in_template.plotItem.setTitle('Average spikes in template {}.  Spike number = {}'.
                                                          format(current_template_index, number_of_spikes))


    def update_marking_led():
        global current_template_index
        if template_marking[current_template_index]:
            label_led_marking.setText('KEPT')
            label_led_marking.setPalette(kept_palette)
        else:
            label_led_marking.setText('DELETED')
            label_led_marking.setPalette(deleted_palette)

    def on_press_button_next():
        global current_template_index
        current_template_index += 1
        update_all_plots()

    def on_press_button_previous():
        global current_template_index
        current_template_index -= 1
        update_all_plots()

    def on_keep():
        global current_template_index
        template_marking[current_template_index] = 1
        update_marking_led()
        np.save(join(base_folder, 'template_marking.npy'), template_marking)

    def on_delete():
        global current_template_index
        template_marking[current_template_index] = 0
        update_marking_led()
        np.save(join(base_folder, 'template_marking.npy'), template_marking)

    def on_slider_update():
        global visibility_threshold
        visibility_threshold = slider_threshold.value() / 10
        update_average_spikes_plot()


    # Main window and layout-----
    app = QtGui.QApplication([])
    main_window = QtGui.QMainWindow()
    main_window.setWindowTitle('Clear kilosort results')
    main_window.resize(800,800)
    central_window = QtGui.QWidget()
    main_window.setCentralWidget(central_window)
    grid_layout = QtGui.QGridLayout()
    central_window.setLayout(grid_layout)
    # ----------------------------

    # Average spikes per template plot
    plot_average_spikes_in_template = pg.PlotWidget(title='Average spikes in template')
    grid_layout.addWidget(plot_average_spikes_in_template, 0, 0, 2, 3)

    electrodes = data.shape[1]
    electrode_curves = []
    for i in range(electrodes):
        electrode_curve = pg.PlotCurveItem()
        plot_average_spikes_in_template.addItem(electrode_curve)
        electrode_curves.append(electrode_curve)
    # ----------------------------

    # Heatmap plot
    #pg.setConfigOption('imageAxisOrder', 'row-major')
    #heatmap_plot = pg.ImageView()
    heatmap_plot = ptl_widget.MatplotlibWidget()
    grid_layout.addWidget(heatmap_plot, 0, 4, 6, 2)
    # ----------------------------

    # Autocorelogram plot
    autocorelogram_plot = pg.PlotWidget(title='Autocorrelogram')
    grid_layout.addWidget(autocorelogram_plot, 2, 0, 2, 3)
    autocorelogram_curve = pg.PlotCurveItem()
    autocorelogram_plot.addItem(autocorelogram_curve)
    # ----------------------------

    # Buttons --------------------
    button_previous = QtGui.QPushButton('Previous')
    button_previous.clicked.connect(on_press_button_previous)
    grid_layout.addWidget(button_previous, 4, 1, 1, 1)

    button_next = QtGui.QPushButton('Next')
    button_next.clicked.connect(on_press_button_next)
    grid_layout.addWidget(button_next, 4, 2, 1, 1)

    button_delete = QtGui.QPushButton('Delete')
    button_delete.clicked.connect(on_delete)
    grid_layout.addWidget(button_delete, 5, 1, 1, 1)

    button_add = QtGui.QPushButton('Keep')
    button_add.clicked.connect(on_keep)
    grid_layout.addWidget(button_add, 5, 2, 1, 1)
    # ----------------------------

    # Slider for visibility threshold
    slider_threshold = QtGui.QSlider(QtCore.Qt.Horizontal)
    slider_threshold.setMinimum(1)
    slider_threshold.setMaximum(40)
    slider_threshold.setTickInterval(1)
    slider_threshold.setTickPosition(QtGui.QSlider.TicksBelow)
    slider_threshold.setSingleStep(1)
    slider_threshold.setValue(int(visibility_threshold*10))
    grid_layout.addWidget(slider_threshold, 6, 0, 1, 3)
    slider_threshold.valueChanged.connect(on_slider_update)
    # ----------------------------

    # LEDs for showing if template is kept or deleted
    label_led_marking = QtGui.QLabel('DELETED')
    grid_layout.addWidget(label_led_marking, 4, 0, 1, 1)
    label_led_marking.setAutoFillBackground(True)
    kept_palette = QtGui.QPalette()
    kept_palette.setColor(QtGui.QPalette.Background, QtCore.Qt.green)
    deleted_palette = QtGui.QPalette()
    deleted_palette.setColor(QtGui.QPalette.Background, QtCore.Qt.red)
    label_led_marking.setPalette(deleted_palette)
    # ----------------------------

    main_window.show()
    update_all_plots()

    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()



def generate_average_over_spikes_per_template(base_folder, binary_data_filename, number_of_channels_in_binary_file,
                                              cut_time_points_around_spike=100):
    channel_map = np.load(join(base_folder, 'channel_map.npy'))
    active_channel_map = np.squeeze(channel_map, axis=1)

    spike_templates = np.load(join(base_folder, r'spike_templates.npy'))
    template_feature_ind = np.load(join(base_folder, 'template_feature_ind.npy'))
    number_of_templates = template_feature_ind.shape[0]

    spike_times = np.squeeze(np.load(join(base_folder, 'spike_times.npy')).astype(np.int))

    num_of_channels = active_channel_map.size

    data_raw = np.memmap(binary_data_filename, dtype=np.int16, mode='r')

    number_of_timepoints_in_raw = int(data_raw.shape[0] / number_of_channels_in_binary_file)
    data_raw_matrix = np.reshape(data_raw, (number_of_channels_in_binary_file, number_of_timepoints_in_raw),
                                 order='F')

    data = np.zeros((number_of_templates, num_of_channels, cut_time_points_around_spike * 2))

    for template in np.arange(number_of_templates):
        spike_indices_in_template = np.argwhere(np.in1d(spike_templates, template))
        spike_times_in_template = np.squeeze(spike_times[spike_indices_in_template])
        num_of_spikes_in_template = spike_indices_in_template.shape[0]
        y = np.zeros((num_of_channels, cut_time_points_around_spike * 2))
        if num_of_spikes_in_template != 0:
            # remove any spikes that don't have enough time points
            too_early_spikes = np.squeeze(np.argwhere(spike_times_in_template < cut_time_points_around_spike), axis=1)
            too_late_spikes = np.squeeze(np.argwhere(spike_times_in_template > number_of_timepoints_in_raw - cut_time_points_around_spike), axis=1)
            out_of_time_spikes = np.concatenate((too_early_spikes, too_late_spikes))
            spike_indices_in_template = np.delete(spike_indices_in_template, out_of_time_spikes)
            num_of_spikes_in_template = spike_indices_in_template.shape[0]

            for spike_in_template in spike_indices_in_template:
                y = y + data_raw_matrix[active_channel_map,
                                        spike_times[spike_in_template] - cut_time_points_around_spike:
                                        spike_times[spike_in_template] + cut_time_points_around_spike]

            y = y / num_of_spikes_in_template
        data[template, :, :] = y
        del y
        print('Added template ' + str(template) + ' with ' + str(num_of_spikes_in_template) + ' spikes')

    np.save(join(base_folder, 'avg_spike_template2.npy'), data)



def _avg_of_single_template(template, 
                               spike_times, 
                               spike_templates,
                               num_of_channels, 
                               cut_time_points_around_spike, 
                               number_of_timepoints_in_raw, 
                               data_raw_matrix,
                               active_channel_map):
    spike_indices_in_template = np.argwhere(np.in1d(spike_templates, template))
    spike_times_in_template = np.squeeze(spike_times[spike_indices_in_template])
    num_of_spikes_in_template = spike_indices_in_template.shape[0]
    y = np.zeros((num_of_channels, cut_time_points_around_spike * 2))
    if num_of_spikes_in_template != 0:
        # remove any spikes that don't have enough time points
        too_early_spikes = np.squeeze(np.argwhere(spike_times_in_template < cut_time_points_around_spike), axis=1)
        too_late_spikes = np.squeeze(np.argwhere(spike_times_in_template > number_of_timepoints_in_raw - cut_time_points_around_spike), axis=1)
        out_of_time_spikes = np.concatenate((too_early_spikes, too_late_spikes))
        spike_indices_in_template = np.delete(spike_indices_in_template, out_of_time_spikes)
        num_of_spikes_in_template = spike_indices_in_template.shape[0]

        for spike_in_template in spike_indices_in_template:
            y = y + data_raw_matrix[active_channel_map,
                                    spike_times[spike_in_template] - cut_time_points_around_spike:
                                    spike_times[spike_in_template] + cut_time_points_around_spike]

        y = y / num_of_spikes_in_template
        print('Added template ' + str(template) + ' with ' + str(num_of_spikes_in_template) + ' spikes')
    return template, y    



def generate_average_over_spikes_per_template_multiprocess(base_folder, binary_data_filename, number_of_channels_in_binary_file,
                                              cut_time_points_around_spike=100):
    channel_map = np.load(join(base_folder, 'channel_map.npy'))
    active_channel_map = np.squeeze(channel_map, axis=1)

    spike_templates = np.load(join(base_folder, r'spike_templates.npy'))
    template_feature_ind = np.load(join(base_folder, 'template_feature_ind.npy'))
    number_of_templates = template_feature_ind.shape[0]

    spike_times = np.squeeze(np.load(join(base_folder, 'spike_times.npy')).astype(np.int))

    num_of_channels = active_channel_map.size

    data_raw = np.memmap(binary_data_filename, dtype=np.int16, mode='r')

    number_of_timepoints_in_raw = int(data_raw.shape[0] / number_of_channels_in_binary_file)
    data_raw_matrix = np.reshape(data_raw, (number_of_channels_in_binary_file, number_of_timepoints_in_raw),
                                 order='F')
    unordered_data = Parallel(n_jobs=8)(delayed(_avg_of_single_template)(i, 
                               spike_times, 
                               spike_templates,
                               num_of_channels, 
                               cut_time_points_around_spike, 
                               number_of_timepoints_in_raw, 
                               data_raw_matrix,
                               active_channel_map) 
    for i in np.arange(number_of_templates))
    data = np.zeros((number_of_templates, num_of_channels, cut_time_points_around_spike * 2))
    for idx, info in unordered_data:
      data[idx,...] = info
      

    np.save(join(base_folder, 'avg_spike_template.npy'), data)



