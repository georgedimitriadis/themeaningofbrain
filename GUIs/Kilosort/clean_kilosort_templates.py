
from __future__ import print_function, absolute_import, division
import numpy as np
from os.path import join, exists
import matplotlib
matplotlib.use('Qt5Agg')
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.widgets import MatplotlibWidget as ptl_widget
from GUIs.Kilosort import spike_heatmap as sh
import pandas as pd
import time

# If the spike trains have more spikes than this value then downsample them to this value by picking that many values
# randomly
MAX_SPIKES_TO_AUTOCORRELATE = 10000

def cleanup_kilosorted_data(base_folder, number_of_channels_in_binary_file, binary_data_filename, prb_file,
                            type_of_binary=np.int16, order_of_binary='F', sampling_frequency=20000,
                            num_of_shanks_for_vis=None):

    spike_templates = np.load(join(base_folder, r'spike_templates.npy'))
    template_feature_ind = np.load(join(base_folder, 'template_feature_ind.npy'))
    number_of_templates = template_feature_ind.shape[0]

    spike_times = np.load(join(base_folder, 'spike_times.npy')).astype(np.int)

    templates = np.load(join(base_folder, 'templates.npy'))

    data_raw = np.memmap(binary_data_filename, dtype=type_of_binary, mode='r')

    number_of_timepoints_in_raw = int(data_raw.shape[0] / number_of_channels_in_binary_file)
    global data_raw_matrix
    data_raw_matrix = np.reshape(data_raw, (number_of_channels_in_binary_file, number_of_timepoints_in_raw),
                                 order=order_of_binary)

    global current_template_index
    current_template_index = 0
    global visibility_threshold
    visibility_threshold = 2
    max_spikes_in_single_spike_window = 4
    global number_of_visible_single_spikes
    number_of_visible_single_spikes = int(max_spikes_in_single_spike_window / 2)


    if exists(join(base_folder, 'template_marking.npy')):
        template_marking = np.load(join(base_folder, 'template_marking.npy'))
    else:
        template_marking = np.zeros(number_of_templates)
        np.save(join(base_folder, 'template_marking.npy'), template_marking)

    global data
    assert exists(join(base_folder, 'avg_spike_template.npy'))
    data = np.load(join(base_folder, 'avg_spike_template.npy'))

    # Update data functions and their helpers
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
        global data
        visible_channels = get_visible_channels(current_template_index=current_template_index,
                                                visibility_threshold=visibility_threshold)
        time_points = data.shape[2]
        total_time = time_points / sampling_frequency
        time_axis = np.arange(-(total_time/2), total_time/2, 1 / sampling_frequency)
        for i in np.arange(electrodes):
            avg_electrode_curves[i].setData(time_axis, data[current_template_index, i, :])
            if i in visible_channels:
                avg_electrode_curves[i].setPen(pg.mkPen((i, number_of_channels_in_binary_file * 1.3)))
            else:
                avg_electrode_curves[i].setPen(pg.mkPen(None))

    def initialize_single_spike_window():
        probe = sh.get_probe_geometry_from_prb_file(prb_file)
        all_electrode_positions = pd.Series(probe[0]['geometry']).tolist()
        all_elec_pos_x = [x for x, y in all_electrode_positions]
        time_points = data.shape[2]
        unique_x_positions = np.unique(all_elec_pos_x)
        x_pos_step = (np.max(all_elec_pos_x) - np.min(all_elec_pos_x)) / (len(unique_x_positions) - 1)
        all_possible_x_positions = np.arange(np.min(unique_x_positions), np.max(unique_x_positions) + x_pos_step,
                                             x_pos_step)
        total_x_axis_points = ((time_points + 20) * len(all_possible_x_positions))

        channel_map = np.squeeze(np.load(join(base_folder, 'channel_map.npy')))
        electrode_positions = pd.Series(probe[0]['geometry'])[channel_map].tolist()
        elec_pos_x = [x for x, y in electrode_positions]
        elec_pos_y = [y for x, y in electrode_positions]
        single_spikes_data = get_all_spikes_form_template_multiprocess()
        y_position_step = np.max(single_spikes_data)

        indices_to_split = np.arange(0, len(channel_map), int(len(channel_map)/20))
        thread_electrodes = np.split(range(len(channel_map)), indices_to_split)
        num_of_spikes = max_spikes_in_single_spike_window

        threads = []
        thread_id = 0
        for electrodes_in_tread in thread_electrodes:
            thread = Thread_initialize_single_spike_plot(electrodes=electrodes_in_tread, num_of_spikes=num_of_spikes,
                                                         total_x_axis_points=total_x_axis_points, time_points=time_points,
                                                         all_possible_x_positions=all_possible_x_positions,
                                                         y_position_step=y_position_step, elec_pos_x=elec_pos_x,
                                                         elec_pos_y=elec_pos_y,
                                                         single_spike_electrode_curves=single_spike_electrode_curves,
                                                         thread_id=thread_id)
            thread_id += 1
            thread.start()
            threads.append(thread)

        for thread in threads:
            while not thread.isFinished():
                time.sleep(0.01)

    class Thread_initialize_single_spike_plot(pg.QtCore.QThread):
        def __init__(self, electrodes, num_of_spikes, total_x_axis_points, time_points, all_possible_x_positions,
                     y_position_step, elec_pos_x, elec_pos_y,
                     single_spike_electrode_curves, thread_id):
            super(Thread_initialize_single_spike_plot, self).__init__()
            self.electrodes = electrodes
            self.num_of_spikes = num_of_spikes
            self.total_x_axis_points = total_x_axis_points
            self.time_points = time_points
            self.all_possible_x_positions = all_possible_x_positions
            self.y_position_step = y_position_step
            self.elec_pos_x = elec_pos_x
            self.elec_pos_y = elec_pos_y
            self.single_spike_electrode_curves = single_spike_electrode_curves
            self.thread_id = thread_id

        def run(self):
            self.initialize_some_electrodes()

        def initialize_some_electrodes(self):
            for electrode in self.electrodes:
                single_spike_curves = []
                #print('Thread id = ' + str(self.thread_id) + ', electrode: ' + str(electrode))
                for spike in range(self.num_of_spikes):
                    #spike_curve = pg.PlotCurveItem()
                    #single_spike_plot.addItem(spike_curve)
                    #single_spike_curves.append(spike_curve)

                    x_position = np.squeeze(np.argwhere(np.in1d(self.all_possible_x_positions, self.elec_pos_x[electrode])))
                    data_to_set = np.empty(self.total_x_axis_points)
                    data_to_set[:] = np.nan
                    data_to_set[(self.time_points + 20) * x_position:(self.time_points + 20) * (x_position + 1) - 20] = \
                        self.y_position_step * self.elec_pos_y[electrode]
                    spike_curve = self.single_spike_electrode_curves[electrode][spike]
                    spike_curve.setData(data_to_set)
                    spike_curve.setPen(pg.mkPen(None))
                    if spike == 0:
                        spike_curve.setPen(pg.mkPen(100, 100, 100, 50))

                single_spike_electrode_curves.append(single_spike_curves)

    def update_single_spikes_plot():
        if not single_spike_window.isVisible():
            return

        global number_of_visible_single_spikes
        global current_template_index
        global visibility_threshold
        visible_electrodes = get_visible_channels(current_template_index=current_template_index,
                                                  visibility_threshold=visibility_threshold)

        probe = sh.get_probe_geometry_from_prb_file(prb_file)
        all_electrode_positions = pd.Series(probe[0]['geometry']).tolist()
        all_elec_pos_x = [x for x, y in all_electrode_positions]
        time_points = data.shape[2]
        unique_x_positions = np.unique(all_elec_pos_x)
        x_pos_step = (np.max(all_elec_pos_x) - np.min(all_elec_pos_x)) / (len(unique_x_positions) - 1)
        all_possible_x_positions = np.arange(np.min(unique_x_positions), np.max(unique_x_positions) + x_pos_step, x_pos_step)
        total_x_axis_points = ((time_points + 20) * len(all_possible_x_positions))

        electrode_positions = pd.Series(probe[0]['geometry'])[visible_electrodes].tolist()
        elec_pos_x = [x for x, y in electrode_positions]
        elec_pos_y = [y for x, y in electrode_positions]

        single_spikes_data = get_all_spikes_form_template_multiprocess()  # shape(spikes, electrodes, time)
        y_position_step = np.max(single_spikes_data)

        num_of_electrodes = single_spikes_data.shape[1]
        print(num_of_electrodes)
        spikes = np.random.choice(range(single_spikes_data.shape[0]), number_of_visible_single_spikes)
        for electrode in range(num_of_electrodes):
            spike_num = 0
            for spike in spikes:

                x_position = np.squeeze(np.argwhere(np.in1d(all_possible_x_positions, elec_pos_x[electrode])))
                data_to_set = np.empty(total_x_axis_points)
                data_to_set[:] = np.nan
                data_to_set[(time_points + 20) * x_position:(time_points + 20) * (x_position + 1) - 20] = \
                    single_spikes_data[spike, electrode, :] + y_position_step * elec_pos_y[electrode]
                spike_curve = single_spike_electrode_curves[electrode][spike_num]
                spike_curve.setData(data_to_set)
                spike_curve.setPen(pg.mkPen((i, number_of_channels_in_binary_file * 1.3)))
                spike_num += 1

    def get_all_spikes_form_template_multiprocess():
        global current_template_index
        global data
        global data_raw_matrix

        visible_channels = get_visible_channels(current_template_index=current_template_index,
                                                visibility_threshold=visibility_threshold)
        time_points = data.shape[2]
        num_of_channels = len(visible_channels)

        spike_indices_in_template = np.argwhere(np.in1d(spike_templates, current_template_index))
        spike_times_in_template = np.squeeze(spike_times[spike_indices_in_template])

        too_early_spikes = np.squeeze(np.argwhere(spike_times_in_template < (time_points / 2)), axis=1)
        too_late_spikes = np.squeeze(
            np.argwhere(spike_times_in_template > number_of_timepoints_in_raw - (time_points / 2)), axis=1)
        out_of_time_spikes = np.concatenate((too_early_spikes, too_late_spikes))
        spike_indices_in_template = np.delete(spike_indices_in_template, out_of_time_spikes)
        num_of_spikes_in_template = spike_indices_in_template.shape[0]

        single_spikes_cube = np.zeros((num_of_spikes_in_template, num_of_channels, time_points))

        num_of_spikes_in_thread = np.ceil(num_of_spikes_in_template / 8)
        starting_spikes = np.concatenate((np.arange(0, num_of_spikes_in_template, num_of_spikes_in_thread),
                                         [num_of_spikes_in_template]))
        end_spikes = starting_spikes[1:]
        starting_spikes = starting_spikes[:-1]
        spike_start_end_indices = np.array((starting_spikes, end_spikes), dtype=np.int32).T

        threads = []
        for start_end_spike in spike_start_end_indices:
            thread = ThreadGetSingleSpikeData(data_raw_matrix, start_end_spike, visible_channels, time_points,
                                              spike_times_in_template, num_of_channels, single_spikes_cube)
            thread.start()
            threads.append(thread)

        for thread in threads:
            while not thread.isFinished():
                time.sleep(0.01)

        return single_spikes_cube

    class ThreadGetSingleSpikeData(pg.QtCore.QThread):
        def __init__(self, data_raw_matrix, start_end_spike, visible_channels, time_points, spike_times_in_template,
                     num_of_channels, single_spikes_cube):
            super(ThreadGetSingleSpikeData, self).__init__()
            self.data_raw_matrix = data_raw_matrix
            self.start_end_spike = start_end_spike
            self.visible_channels = visible_channels
            self.time_points = time_points
            self.spike_times_in_template = spike_times_in_template
            self.num_of_channels = num_of_channels
            self.single_spikes_cube = single_spikes_cube

        def run(self):
            self.single_spikes_cube[self.start_end_spike[0]:self.start_end_spike[1], :, :] = self.get_some_spikes_from_template()

        def get_some_spikes_from_template(self):
            spike_times_in_thread = self.spike_times_in_template[self.start_end_spike[0]:self.start_end_spike[1]]
            single_spikes_cube = np.zeros((len(spike_times_in_thread), self.num_of_channels, self.time_points))
            single_spike_index = 0
            for spike in spike_times_in_thread:
                single_spikes_cube[single_spike_index, :, :] = self.data_raw_matrix[self.visible_channels,
                                                                    int(spike - (self.time_points / 2)):
                                                                    int(spike + (self.time_points / 2))]
                single_spike_index += 1

            return single_spikes_cube

    def update_heatmap_plot():
        connected = np.squeeze(np.load(join(base_folder, 'channel_map.npy')))
        connected_binary = np.in1d(np.arange(number_of_channels_in_binary_file), connected)
        bad_channels = np.squeeze(np.argwhere(connected_binary == False).astype(np.int))
        #bad_channels = None
        sh.create_heatmap_on_matplotlib_widget(heatmap_plot, data[current_template_index], prb_file, window_size=60,
                                               bad_channels=bad_channels, num_of_shanks=num_of_shanks_for_vis,
                                               rotate_90=True, flip_ud=False, flip_lr=False)
        heatmap_plot.draw()

    def update_autocorelogram():
        global current_template_index
        spike_indices_in_template = np.argwhere(np.in1d(spike_templates, current_template_index))
        spike_indices_to_autocorrelate = np.squeeze(spike_indices_in_template)
        if spike_indices_to_autocorrelate.size > MAX_SPIKES_TO_AUTOCORRELATE:
            spike_indices_to_autocorrelate = np.random.choice(spike_indices_to_autocorrelate, MAX_SPIKES_TO_AUTOCORRELATE)
        diffs, norm = crosscorrelate_spike_trains(spike_times[spike_indices_to_autocorrelate].astype(np.int64),
                                                  spike_times[spike_indices_to_autocorrelate].astype(np.int64),
                                                  lag=3000)
        hist, edges = np.histogram(diffs, bins=100)
        autocorelogram_curve.setData(x=edges, y=hist, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))

        number_of_spikes = len(spike_indices_in_template)
        num_of_kept_templates = int(np.sum(template_marking > 0))
        plot_average_spikes_in_template.plotItem.setTitle('Average spikes in template {}.  Spike number = {}\n Kept templates = {}'.
                                                          format(current_template_index, number_of_spikes, num_of_kept_templates))

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

    def update_marking_led():
        global current_template_index
        if template_marking[current_template_index]== 1:
            label_led_marking.setText('Single Unit')
            label_led_marking.setPalette(su_palette)
        elif template_marking[current_template_index]== 2:
            label_led_marking.setText('SU Contaminated')
            label_led_marking.setPalette(suc_palette)
        elif template_marking[current_template_index]== 3:
            label_led_marking.setText('SU Putative')
            label_led_marking.setPalette(sup_palette)
        elif template_marking[current_template_index]== 4:
            label_led_marking.setText('Multi Unit')
            label_led_marking.setPalette(mu_palette)
        elif template_marking[current_template_index]== 5:
            label_led_marking.setText('Unclasified 1')
            label_led_marking.setPalette(un1_palette)
        elif template_marking[current_template_index]== 6:
            label_led_marking.setText('Unclasified 2')
            label_led_marking.setPalette(un2_palette)
        elif template_marking[current_template_index]== 7:
            label_led_marking.setText('Unclasified 3')
            label_led_marking.setPalette(un3_palette)
        else:
            label_led_marking.setText('Noise')
            label_led_marking.setPalette(noise_palette)
    # ----------------------------

    # On_do_something functions
    def on_press_button_next():
        global current_template_index
        current_template_index += 1
        if current_template_index > number_of_templates - 1:
            return
        spike_indices_in_template = np.argwhere(np.in1d(spike_templates, current_template_index))
        number_of_spikes = len(spike_indices_in_template)
        while number_of_spikes == 0:
            current_template_index += 1
            if current_template_index > number_of_templates - 1:
                return
            spike_indices_in_template = np.argwhere(np.in1d(spike_templates, current_template_index))
            number_of_spikes = len(spike_indices_in_template)
        update_all_plots()

    def on_press_button_previous():
        global current_template_index
        current_template_index -= 1
        if current_template_index < 0:
            return
        spike_indices_in_template = np.argwhere(np.in1d(spike_templates, current_template_index))
        number_of_spikes = len(spike_indices_in_template)
        while number_of_spikes == 0:
            current_template_index -= 1
            if current_template_index < 0:
                return
            spike_indices_in_template = np.argwhere(np.in1d(spike_templates, current_template_index))
            number_of_spikes = len(spike_indices_in_template)
        update_all_plots()

    #def on_classify_combo_box_index_change(index):
    #    global current_template_index
    #    template_marking[current_template_index] = index
    #    update_marking_led()
    #    np.save(join(base_folder, 'template_marking.npy'), template_marking)

    def _classify_template(index):
        global current_template_index
        template_marking[current_template_index] = index
        update_marking_led()
        np.save(join(base_folder, 'template_marking.npy'), template_marking)

    def on_press_classify_button_su():
        _classify_template(1)

    def on_press_classify_button_suc():
        _classify_template(2)

    def on_press_classify_button_sup():
        _classify_template(3)

    def on_press_classify_button_mu():
        _classify_template(4)

    def on_press_classify_button_noise():
        _classify_template(0)

    def on_press_classify_button_unc1():
        _classify_template(5)

    def on_press_classify_button_unc2():
        _classify_template(6)

    def on_press_classify_button_unc3():
        _classify_template(7)

    def on_visible_electrodes_slider_update():
        global visibility_threshold
        visibility_threshold = slider_visible_electrodes_threshold.value() / 10
        update_average_spikes_plot()
        update_single_spikes_plot()

    def on_visible_spikes_slider_update():
        global number_of_visible_single_spikes
        number_of_visible_single_spikes = slider_visible_spikes_threshold.value()
        update_single_spikes_plot()

    def on_show_single_spikes():
        if not single_spike_window.isVisible():
            initialize_single_spike_window()
            single_spike_window.show()
        update_single_spikes_plot()

    def on_template_dropdown_changed():
        global current_template_index
        current_template_index = template_dropdown_list.currentRow()
        update_all_plots()
    # ----------------------------

    # Main window and layout-----
    app = QtGui.QApplication([])
    main_window = QtGui.QMainWindow()
    main_window.setWindowTitle('Clear kilosort results')
    main_window.resize(1300, 1000)
    central_window = QtGui.QWidget()
    main_window.setCentralWidget(central_window)
    grid_layout = QtGui.QGridLayout()
    central_window.setLayout(grid_layout)
    # ----------------------------

    # Average spikes per template plot
    plot_average_spikes_in_template = pg.PlotWidget(title='Average spikes in template')
    grid_layout.addWidget(plot_average_spikes_in_template, 0, 0, 2, 5)

    electrodes = data.shape[1]
    avg_electrode_curves = []
    for i in range(electrodes):
        electrode_curve = pg.PlotCurveItem()
        plot_average_spikes_in_template.addItem(electrode_curve)
        avg_electrode_curves.append(electrode_curve)
    # ----------------------------

    # Heatmap plot ---------------
    heatmap_plot = ptl_widget.MatplotlibWidget()
    grid_layout.addWidget(heatmap_plot, 0, 5, 11, 3)
    # ----------------------------

    # Autocorelogram plot --------
    autocorelogram_plot = pg.PlotWidget(title='Autocorrelogram')
    grid_layout.addWidget(autocorelogram_plot, 3, 0, 2, 5)
    autocorelogram_curve = pg.PlotCurveItem()
    autocorelogram_plot.addItem(autocorelogram_curve)
    # ----------------------------

    # Single Spikes Window -------
    single_spike_window = QtGui.QWidget()
    single_spike_window.resize(800, 1000)
    single_spike_layout = QtGui.QGridLayout()
    single_spike_window.setLayout(single_spike_layout)
    single_spike_plot = pg.PlotWidget()
    single_spike_layout.addWidget(single_spike_plot, 0, 0, 2, 5)
    single_spike_electrode_curves = []
    '''
    total_curves = electrodes * max_spikes_in_single_spike_window
    current_curve = 0
    for elec in range(electrodes):
        single_spike_curves = []
        for spike in range(max_spikes_in_single_spike_window):
            electrode_spike_curve = pg.PlotCurveItem()
            single_spike_plot.addItem(electrode_spike_curve)
            single_spike_curves.append(electrode_spike_curve)
            current_curve += 1
            if ((current_curve - 1) / total_curves) % 0.1 > 0.05 and (current_curve / total_curves) % 0.1 < 0.01:
                print('Finished preparing ' + str(int(100 * current_curve / total_curves)) + '%')
        single_spike_electrode_curves.append(single_spike_curves)
    '''
    # ----------------------------

    # Buttons --------------------
    button_previous = QtGui.QPushButton('Previous')
    button_previous.clicked.connect(on_press_button_previous)
    button_previous.setMaximumHeight(500)
    grid_layout.addWidget(button_previous, 5, 1, 2, 1)

    button_next = QtGui.QPushButton('Next')
    button_next.clicked.connect(on_press_button_next)
    button_next.setMaximumHeight(500)
    grid_layout.addWidget(button_next, 5, 2, 2, 1)

    button_su = QtGui.QPushButton('Single Unit')
    button_su.clicked.connect(on_press_classify_button_su)
    button_su.setStyleSheet("background-color: green")
    button_su.setMaximumHeight(500)
    grid_layout.addWidget(button_su, 7, 1, 2, 1)

    button_suc = QtGui.QPushButton('S.U. Contaminated')
    button_suc.clicked.connect(on_press_classify_button_suc)
    button_suc.setStyleSheet("background-color: blue")
    button_suc.setMaximumHeight(500)
    grid_layout.addWidget(button_suc, 7, 2, 2, 1)

    button_sup = QtGui.QPushButton('S.U. Putative')
    button_sup.clicked.connect(on_press_classify_button_sup)
    button_sup.setStyleSheet("background-color: darkBlue")
    button_sup.setMaximumHeight(500)
    grid_layout.addWidget(button_sup, 7, 3, 2, 1)

    button_mu = QtGui.QPushButton('Multi Unit')
    button_mu.clicked.connect(on_press_classify_button_mu)
    button_mu.setStyleSheet("background-color: magenta")
    button_mu.setMaximumHeight(500)
    grid_layout.addWidget(button_mu, 7, 4, 2, 1)

    button_noise = QtGui.QPushButton('NOISE')
    button_noise.clicked.connect(on_press_classify_button_noise)
    button_noise.setStyleSheet("background-color: red")
    button_noise.setMaximumHeight(500)
    grid_layout.addWidget(button_noise, 9, 1, 2, 1)

    button_unc1 = QtGui.QPushButton('Unclassified 1')
    button_unc1.clicked.connect(on_press_classify_button_unc1)
    button_unc1.setStyleSheet("background-color: darkGray")
    button_unc1.setMaximumHeight(500)
    grid_layout.addWidget(button_unc1, 9, 2, 2, 1)

    button_unc2 = QtGui.QPushButton('Unclassified 2')
    button_unc2.clicked.connect(on_press_classify_button_unc2)
    button_unc2.setStyleSheet("background-color: gray")
    button_unc2.setMaximumHeight(500)
    grid_layout.addWidget(button_unc2, 9, 3, 2, 1)

    button_unc3 = QtGui.QPushButton('Unclassified 3')
    button_unc3.clicked.connect(on_press_classify_button_unc3)
    button_unc3.setStyleSheet("background-color: darkGray")
    button_unc3.setMaximumHeight(500)
    grid_layout.addWidget(button_unc3, 9, 4, 2, 1)

    #classify_combo_box = QtGui.QComboBox()
    #classify_combo_box.addItem('Noise')
    #classify_combo_box.addItem('Single Unit')
    #classify_combo_box.addItem('Single Unit Contaminated')
    #classify_combo_box.addItem('Single Unit Putative')
    #classify_combo_box.addItem('Multi Unit')
    #classify_combo_box.addItem('Unclassified 1')
    #classify_combo_box.addItem('Unclassified 2')
    #classify_combo_box.addItem('Unclassified 3')
    #classify_combo_box.activated.connect(on_classify_combo_box_index_change)
    #grid_layout.addWidget(classify_combo_box, 6, 1, 1, 2)


    button_show_single_spikes = QtGui.QPushButton('Show single spikes')
    button_show_single_spikes.clicked.connect(on_show_single_spikes)
    button_show_single_spikes.setMaximumHeight(500)
    grid_layout.addWidget(button_show_single_spikes, 5, 4, 2, 1)
    # ----------------------------

    # Slider for electrode visibility threshold
    slider_visible_electrodes_threshold = QtGui.QSlider(QtCore.Qt.Horizontal)
    slider_visible_electrodes_threshold.setMinimum(1)
    slider_visible_electrodes_threshold.setMaximum(40)
    slider_visible_electrodes_threshold.setTickInterval(1)
    slider_visible_electrodes_threshold.setTickPosition(QtGui.QSlider.TicksBelow)
    slider_visible_electrodes_threshold.setSingleStep(1)
    slider_visible_electrodes_threshold.setValue(int(visibility_threshold*10))
    grid_layout.addWidget(slider_visible_electrodes_threshold, 2, 0, 1, 3)
    slider_visible_electrodes_threshold.sliderReleased.connect(on_visible_electrodes_slider_update)
    # ----------------------------

    # Slider for number of single spikes visible threshold
    slider_visible_spikes_threshold = QtGui.QSlider(QtCore.Qt.Horizontal)
    slider_visible_spikes_threshold.setMinimum(1)
    slider_visible_spikes_threshold.setMaximum(max_spikes_in_single_spike_window)
    slider_visible_spikes_threshold.setTickInterval(5)
    slider_visible_spikes_threshold.setTickPosition(QtGui.QSlider.TicksBelow)
    slider_visible_spikes_threshold.setSingleStep(5)
    slider_visible_spikes_threshold.setValue(int(number_of_visible_single_spikes))
    single_spike_layout.addWidget(slider_visible_spikes_threshold, 3, 0, 1, 5)
    slider_visible_spikes_threshold.sliderReleased.connect(on_visible_spikes_slider_update)
    # ----------------------------

    # List for selecting a template
    template_dropdown_list = QtGui.QListWidget()
    template_dropdown_list.setMaximumHeight(1000)
    template_dropdown_list.setMaximumWidth(300)
    grid_layout.addWidget(template_dropdown_list, 5, 0, 5, 1)
    template_dropdown_list.SingleSelection
    for t in range(number_of_templates):
        item = QtGui.QListWidgetItem('Template number ' + str(t))
        template_dropdown_list.addItem(item)
    template_dropdown_list.itemSelectionChanged.connect(on_template_dropdown_changed)
    # ----------------------------

    # LEDs for showing if template is kept or deleted
    label_led_marking = QtGui.QLabel('Noise')
    grid_layout.addWidget(label_led_marking, 5, 3, 2, 1)
    label_led_marking.setMaximumHeight(500)
    label_led_marking.setAutoFillBackground(True)
    su_palette = QtGui.QPalette()
    su_palette.setColor(QtGui.QPalette.Background, QtCore.Qt.green)
    suc_palette = QtGui.QPalette()
    suc_palette.setColor(QtGui.QPalette.Background, QtCore.Qt.blue)
    sup_palette = QtGui.QPalette()
    sup_palette.setColor(QtGui.QPalette.Background, QtCore.Qt.darkBlue)
    mu_palette = QtGui.QPalette()
    mu_palette.setColor(QtGui.QPalette.Background, QtCore.Qt.magenta)
    un1_palette = QtGui.QPalette()
    un1_palette.setColor(QtGui.QPalette.Background, QtCore.Qt.darkGray)
    un2_palette = QtGui.QPalette()
    un2_palette.setColor(QtGui.QPalette.Background, QtCore.Qt.gray)
    un3_palette = QtGui.QPalette()
    un3_palette.setColor(QtGui.QPalette.Background, QtCore.Qt.darkGray)
    noise_palette = QtGui.QPalette()
    noise_palette.setColor(QtGui.QPalette.Background, QtCore.Qt.red)
    label_led_marking.setPalette(noise_palette)
    # ----------------------------

    main_window.show()
    update_all_plots()

    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit(app.exec_())




