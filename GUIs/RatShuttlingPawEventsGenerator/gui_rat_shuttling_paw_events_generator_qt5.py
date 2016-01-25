__author__ = 'George Dimitriadis'


import sys
import os.path
import cv2

from PyQt5 import QtCore, QtWidgets

import matplotlib.cm as cm
import matplotlib.text as mpt
import timelocked_analysis_functions as tlf
import numpy as np
import pandas as pd
import time

from ui_rat_shuttling_paw_events_generator_qt5 import Ui_RatShuttlingPawEventsGenerator


# Thread to run the video in (controlled by the on_pB_Run_toggled slot)
class RunVideoThread(QtCore.QThread):
    def __init__(self, _ui):
        QtCore.QThread.__init__(self)
        self._ui = _ui

    run_video = False

    def run(self):
        i = self._ui.sBox_FrameNum.value()
        step = self._ui.sBox_frames_step.value()
        sleep_time = 1 / self._ui.sB_updates_sec.value()
        while self.run_video:
            i = i + step
            self._ui.sBox_FrameNum.setValue(i)
            time.sleep(sleep_time)
        return




# Main class
class RatShuttlingPawEventsGenerator:

    def __init__(self):
        # paths
        self.front_video_path = r"\front_video.avi"
        self.front_counter_path = r"\front_counter.csv"
        self.top_video_path = r"\top_video.avi"
        self.adc_path = r"\adc.bin"
        self.sync_path = r"\sync.bin"
        self.analysis_folder = "Analysis"
        self.analysis_file_name = r"\session.hdf5"

        # session.hdf5 structure
        self.fronttime_key = 'video/front/time'
        self.fronttrials_key = 'video/front/trials'
        self.toptime_key = 'video/top/time'
        self.paw_events_key = 'task/events/paws'
        self.good_trials_key = 'task/events/trials'
        self.trajectories_key = 'task/trajectories'

        # colums of trial_start_stop_info in session.hdf5
        self.trials_info_start_frame = "start frame"
        self.trials_info_stop_frame = "stop frame"
        self.trials_info_start_frame_time = "start frame time"
        self.trials_info_end_frame_time = "end frame time"
        self.trials_info_trial_duration = "trial duration"

        # colums of paw_events in session.hdf5
        self.blpaw = 'back left paw'
        self.brpaw = 'back right paw'
        self.flpaw = 'front left paw'
        self.frpaw = 'front right paw'
        self.trial_paw_event = 'trial of event'
        self.time_paw_event = 'time of event'

        # columns of trajectories in session.hdf5
        self.name_traj_point = 'name of trajectory point'
        self.trial_traj_point = 'trial of trajectory point'
        self.frame_traj_point = 'frame of trajectory point'
        self.time_traj_point = 'time of trajectory point'
        self.x_traj_point = 'X of trajectory point'
        self.y_traj_point = 'Y of trajectory point'

        # instance variables
        app = QtWidgets.QApplication(sys.argv)
        window = QtWidgets.QMainWindow()
        self.ui = Ui_RatShuttlingPawEventsGenerator()
        self.ui.setupUi(window)
        self.rec_freq = 8000
        self.ss_freq = 100
        self.session = ""
        self.front_video = ""
        self.top_video = ""
        self.corrected_frame_numbers = []
        self.cam_shutter_closing_samples = []
        self.analysis_exists = False
        self.data_loaded = False
        self.t = RunVideoThread(self.ui)
        self.trajectory_lines = []
        self.annotation = mpt.Annotation("", xy=(-1,-1))

        self.connect_slots()

        window.show()
        app.exec_()

    # Slot that deals with the initialization of the GUI once data are loaded
    @QtCore.pyqtSlot('QString')
    def on_pB_select_data_clicked(self):

        self.ui.label_data_loaded.setText("Loading\nData")

        self.session = QtWidgets.QFileDialog.getExistingDirectory(parent=None, caption="Select Data Directory")
        self.ui.cB_trial_start_frames.clear()
        if os.path.isfile(self.session+self.front_video_path):
            self.front_video = cv2.VideoCapture(self.session + self.front_video_path)

            self.adc = tlf.load_raw_data(self.session + self.adc_path, numchannels=8, dtype=np.uint16).dataMatrix
            self.adc_ss = tlf.subsample_basis_data(self.adc, self.rec_freq, self.ss_freq, 'fir', 29)

            front_counter = tlf.load_colum_from_csv(self.session + self.front_counter_path, 0)
            self.corrected_frame_numbers = front_counter - front_counter[0]

            sync = np.squeeze(tlf.load_raw_data(self.session + self.sync_path, numchannels=1, dtype=np.uint8).dataMatrix)
            sync_diff = np.diff(sync.astype('int8'))
            self.cam_shutter_closing_samples = np.squeeze((sync_diff < -0.9).nonzero())

            self.ui.label_data_loaded.setText("Data\nLoaded")
            self.data_loaded = True

            analysis_path = os.path.join(self.session, self.analysis_folder)
            if os.path.exists(analysis_path):  # Add the list of good trials to the combobox
                self.analysis_exists = True
                session_hdf5 = pd.HDFStore(analysis_path+self.analysis_file_name)
                trials_start_stop_info = session_hdf5[self.fronttrials_key]
                start_frames_strlist =np.char.mod('%d', trials_start_stop_info[self.trials_info_start_frame].tolist())
                self.ui.cB_trial_start_frames.addItems(start_frames_strlist)
                if self.paw_events_key in session_hdf5:  # Add the lists of paw touch events to their comboboxes
                    paw_events = session_hdf5[self.paw_events_key]; """:type : pd.DataFrame"""
                    self.ui.cB_br_paw_frames.addItems([str(x) for x in paw_events[paw_events[self.brpaw] != -1][self.brpaw]])
                    self.ui.cB_bl_paw_frames.addItems([str(x) for x in paw_events[paw_events[self.blpaw] != -1][self.blpaw]])
                    self.ui.cB_fr_paw_frames.addItems([str(x) for x in paw_events[paw_events[self.frpaw] != -1][self.frpaw]])
                    self.ui.cB_fl_paw_frames.addItems([str(x) for x in paw_events[paw_events[self.flpaw] != -1][self.flpaw]])
                if self.good_trials_key in session_hdf5:  # Add the list of good_tirls to its combobox
                    good_trials = session_hdf5[self.good_trials_key]; """:type : pd.Series"""
                    self.ui.cB_selected_trials.addItems([str(x) for x in good_trials])
                if self.trajectories_key in session_hdf5:
                    trajectories = session_hdf5[self.trajectories_key]; """:type : pd.DataFrame"""
                    self.ui.cB_trajectory_name.addItems(list(set(trajectories[self.name_traj_point].tolist())))
                session_hdf5.close()
        else:
            self.ui.label_data_loaded.setText("No Data\nLoaded")
            self.data_loaded = False


    @QtCore.pyqtSlot(int)
    def on_sBox_FrameNum_valueChanged(self,i):
        if self.data_loaded:
            self.front_video.set(cv2.CAP_PROP_POS_FRAMES, i)
            r, f = self.front_video.read()
            if not r:
                self.ui.sBox_FrameNum.setValue(0)
                self.front_video.set(cv2.CAP_PROP_POS_FRAMES, i)
                r, f = self.front_video.read()
            resize_factor = self.ui.dSB_frame_resize.value()
            half_f = cv2.resize(f, dsize=(0, 0), fx=resize_factor, fy=resize_factor, interpolation = cv2.INTER_AREA)
            self.ui.mplg_frames._dataY = half_f
            self.ui.mplg_frames.all_sp_axes[0].clear()
            self.ui.mplg_frames.all_sp_axes[0].imshow(self.ui.mplg_frames._dataY, cmap=cm.gray)
            self.ui.mplg_frames.all_sp_axes[0].axis('image')
            self.trajectory_lines = []
            self.ui.mplg_frames.canvas.draw()

            self.draw_piezos(i)
            self.ui.label_trial_number_int.setText(str(self.get_trial_num(i)))

    def get_trial_num(self, frame_num):
        if self.analysis_exists:
            trials_start_frames = self.get_all_combobox_values(self.ui.cB_trial_start_frames)
            trial_num_list = [i for i, x in enumerate(trials_start_frames[:-1]) if int(x) <= frame_num and int(trials_start_frames[i+1]) > frame_num]
            if trial_num_list:
                trial_num = trial_num_list[0]
            else:
                trial_num = np.size(trials_start_frames)
            return trial_num
        else:
            return -1


    def draw_piezos(self, frame_num):
        sync_pulse_index = self.corrected_frame_numbers[frame_num]
        sample_of_frame = self.cam_shutter_closing_samples[sync_pulse_index]
        ss_sample_of_frame = int(sample_of_frame*float(self.ss_freq/self.rec_freq))

        piezo_samples_to_plot = self.ui.hSB_piezo_samples_to_plot.value()
        piezo_time = np.arange(-piezo_samples_to_plot/2,piezo_samples_to_plot/2)
        for k in np.arange(0, 8):
            data = self.adc_ss[k, ss_sample_of_frame-(piezo_samples_to_plot/2):ss_sample_of_frame+(piezo_samples_to_plot/2)]
            self.ui.mplg_piezos.all_sp_axes[k].clear()
            self.ui.mplg_piezos.all_sp_axes[k].plot(piezo_time, data)
        self.ui.mplg_piezos.canvas.draw()


    @QtCore.pyqtSlot(int)
    def on_hSB_piezo_samples_to_plot_valueChanged(self, i):
        if self.data_loaded:
            frame = self.ui.sBox_FrameNum.value()
            self.draw_piezos(frame)


    @QtCore.pyqtSlot(int)
    def on_sBox_frames_step_valueChanged(self, i):
        self.ui.sBox_FrameNum.setSingleStep(i)


    @QtCore.pyqtSlot(bool)
    def on_pB_Run_toggled(self, state):
        if self.data_loaded:
            if state:
                self.t.run_video = True
                if not self.t.isRunning():
                    self.t.start()
            else:
                self.t.run_video = False

    # Slot to resize the image of the video
    @QtCore.pyqtSlot(int)
    def on_dSB_frame_resize_valueChanged(self):
        if self.data_loaded:
            self.front_video.set(cv2.CAP_PROP_POS_FRAMES, 1)
            r, f = self.front_video.read()
            resize_factor = self.ui.dSB_frame_resize.value()
            half_f = cv2.resize(f, dsize=(0, 0), fx=resize_factor, fy=resize_factor, interpolation = cv2.INTER_AREA)
            image_size = np.max(np.shape(half_f))
            if image_size >= 1000:
                self.ui.sB_updates_sec.setMaximum(1)
            elif image_size<1000 and image_size >= 500:
                self.ui.sB_updates_sec.setMaximum(2)
            elif image_size < 500 and image_size >=250:
                self.ui.sB_updates_sec.setMaximum(3)
            else:
                self.ui.sB_updates_sec.setMaximum(5)

    # Slot to move the video to the current frame shown on the trials start frames combobox
    @QtCore.pyqtSlot(bool)
    def on_pB_goto_trial_frame_clicked(self):
        if self.data_loaded:
            frame = int(self.ui.cB_trial_start_frames.currentText())
            if not np.isnan(frame):
                self.ui.sBox_FrameNum.setValue(frame)

    # Slots to add and remove the good trials to session.hdf5
    @QtCore.pyqtSlot(bool)
    def on_pB_add_trial_clicked(self):
        if self.data_loaded:
            trial = int(self.ui.label_trial_number_int.text())
            all_trials = [int(x) for x in self.get_all_combobox_values(self.ui.cB_selected_trials)]
            if trial != -1 and trial not in all_trials:
                self.ui.cB_selected_trials.addItem(str(trial))

                if self.analysis_exists:
                    analysis_path = os.path.join(self.session, self.analysis_folder)
                    session_hdf5 = pd.HDFStore(analysis_path+self.analysis_file_name)
                    if self.good_trials_key in session_hdf5:
                        good_trials = session_hdf5[self.good_trials_key]; """:type : pd.Series"""
                        good_trials.set_value(max(good_trials.keys())+1, trial)
                    else:
                        good_trials = pd.Series(trial)
                    good_trials.to_hdf(analysis_path+self.analysis_file_name, self.good_trials_key)
                    session_hdf5.close()


    @QtCore.pyqtSlot(bool)
    def on_pB_remove_trial_clicked(self):
        if self.data_loaded and self.ok_dialog():
            index =self.ui.cB_selected_trials.currentIndex()
            trial = int(self.ui.cB_selected_trials.currentText())
            self.ui.cB_selected_trials.removeItem(index)

            if self.analysis_exists:
                analysis_path = os.path.join(self.session, self.analysis_folder)
                session_hdf5 = pd.HDFStore(analysis_path+self.analysis_file_name)
                if self.good_trials_key in session_hdf5:
                    good_trials = session_hdf5[self.good_trials_key]; """:type : pd.Series"""
                    good_trials = good_trials[good_trials != trial]
                    good_trials.reset_index(drop=True)
                    good_trials.to_hdf(analysis_path+self.analysis_file_name, self.good_trials_key)
                session_hdf5.close()


    # Slots for adding and removing paw touching events to session.hdf5
    @QtCore.pyqtSlot(bool)
    def on_pB_fl_paw_add_frame_clicked(self):
        if self.data_loaded and self.analysis_exists:
            frame = self.ui.sBox_FrameNum.value()
            all_frames = self.get_all_combobox_values(self.ui.cB_fl_paw_frames)
            if frame not in all_frames:
                self.ui.cB_fl_paw_frames.addItem(str(frame))
                self.ui.cB_fl_paw_frames.setCurrentIndex(self.ui.cB_fl_paw_frames.count()-1)
                self.update_paw_events_dataframe(self.flpaw)

    @QtCore.pyqtSlot(bool)
    def on_pB_fl_paw_remove_frame_clicked(self):
        if self.data_loaded and self.analysis_exists and self.ok_dialog():
            self.ui.cB_fl_paw_frames.removeItem(self.ui.cB_fl_paw_frames.currentIndex())
            analysis_path = os.path.join(self.session, self.analysis_folder)
            session_hdf5 = pd.HDFStore(analysis_path+self.analysis_file_name)
            if self.paw_events_key in session_hdf5:
                paw_events = session_hdf5[self.paw_events_key]; """:type : pd.DataFrame"""
                paw_events = paw_events[paw_events[self.flpaw] != int(self.ui.cB_fl_paw_frames.currentText())]
                paw_events.to_hdf(analysis_path+self.analysis_file_name, self.paw_events_key)
                session_hdf5.close()

    @QtCore.pyqtSlot(bool)
    def on_pB_fr_paw_add_frame_clicked(self):
        if self.data_loaded and self.analysis_exists:
            frame = self.ui.sBox_FrameNum.value()
            all_frames = self.get_all_combobox_values(self.ui.cB_fr_paw_frames)
            if frame not in all_frames:
                self.ui.cB_fr_paw_frames.addItem(str(frame))
                self.ui.cB_fr_paw_frames.setCurrentIndex(self.ui.cB_fr_paw_frames.count()-1)
                self.update_paw_events_dataframe(self.frpaw)

    @QtCore.pyqtSlot(bool)
    def on_pB_fr_paw_remove_frame_clicked(self):
        if self.data_loaded and self.analysis_exists and self.ok_dialog():
            self.ui.cB_fr_paw_frames.removeItem(self.ui.cB_fr_paw_frames.currentIndex())
            analysis_path = os.path.join(self.session, self.analysis_folder)
            session_hdf5 = pd.HDFStore(analysis_path+self.analysis_file_name)
            if self.paw_events_key in session_hdf5:
                paw_events = session_hdf5[self.paw_events_key]; """:type : pd.DataFrame"""
                paw_events = paw_events[paw_events[self.frpaw] != int(self.ui.cB_fr_paw_frames.currentText())]
                paw_events.to_hdf(analysis_path+self.analysis_file_name, self.paw_events_key)
                session_hdf5.close()

    @QtCore.pyqtSlot(bool)
    def on_pB_bl_paw_add_frame_clicked(self):
        if self.data_loaded and self.analysis_exists:
            frame = self.ui.sBox_FrameNum.value()
            all_frames = self.get_all_combobox_values(self.ui.cB_bl_paw_frames)
            if frame not in all_frames:
                self.ui.cB_bl_paw_frames.addItem(str(frame))
                self.ui.cB_bl_paw_frames.setCurrentIndex(self.ui.cB_bl_paw_frames.count()-1)
                self.update_paw_events_dataframe(self.blpaw)

    @QtCore.pyqtSlot(bool)
    def on_pB_bl_paw_remove_frame_clicked(self):
        if self.data_loaded and self.analysis_exists and self.ok_dialog():
            self.ui.cB_bl_paw_frames.removeItem(self.ui.cB_bl_paw_frames.currentIndex())

            analysis_path = os.path.join(self.session, self.analysis_folder)
            session_hdf5 = pd.HDFStore(analysis_path+self.analysis_file_name)
            if self.paw_events_key in session_hdf5:
                paw_events = self.session_hdf5[self.paw_events_key]; """:type : pd.DataFrame"""
                paw_events = paw_events[paw_events[self.blpaw] != int(self.ui.cB_bl_paw_frames.currentText())]
                paw_events.to_hdf(analysis_path+self.analysis_file_name, self.paw_events_key)
                session_hdf5.close()

    @QtCore.pyqtSlot(bool)
    def on_pB_br_paw_add_frame_clicked(self):
        if self.data_loaded and self.analysis_exists:
            frame = self.ui.sBox_FrameNum.value()
            all_frames = self.get_all_combobox_values(self.ui.cB_br_paw_frames)
            if frame not in all_frames:
                self.ui.cB_br_paw_frames.addItem(str(frame))
                self.ui.cB_br_paw_frames.setCurrentIndex(self.ui.cB_br_paw_frames.count()-1)
                self.update_paw_events_dataframe(self.brpaw)

    @QtCore.pyqtSlot(bool)
    def on_pB_br_paw_remove_frame_clicked(self):
        if self.data_loaded and self.analysis_exists and self.ok_dialog():

            self.ui.cB_br_paw_frames.removeItem(self.ui.cB_br_paw_frames.currentIndex())

            analysis_path = os.path.join(self.session, self.analysis_folder)
            session_hdf5 = pd.HDFStore(analysis_path+self.analysis_file_name)
            if self.paw_events_key in session_hdf5:
                paw_events = session_hdf5[self.paw_events_key]; """:type : pd.DataFrame"""
                paw_events = paw_events[paw_events[self.brpaw] != int(self.ui.cB_br_paw_frames.currentText())]
                paw_events.to_hdf(analysis_path+self.analysis_file_name, self.paw_events_key)
                session_hdf5.close()


    def update_paw_events_dataframe(self, paw):
        frame_num = self.ui.sBox_FrameNum.value()

        analysis_path = os.path.join(self.session, self.analysis_folder)
        session_hdf5 = pd.HDFStore(analysis_path+self.analysis_file_name)
        time_of_frame = session_hdf5[self.fronttime_key].iloc[frame_num]

        trial_num = int(self.ui.label_trial_number_int.text())
        self.ui.label_trial_number_int.setText(str(trial_num))

        if self.paw_events_key in session_hdf5:
            paw_events = session_hdf5[self.paw_events_key]; """:type : pd.DataFrame"""
        else:
            paw_events = pd.DataFrame(columns=[self.trial_paw_event, self.time_paw_event,
                                               self.blpaw, self.brpaw,
                                               self.flpaw, self.frpaw])

        def bl_paw():
            return np.array([[trial_num, time_of_frame,
                              frame_num, -1, -1, -1]])

        def br_paw():
            return np.array([[trial_num, time_of_frame,
                              -1, frame_num, -1, -1]])

        def fl_paw():
            return np.array([[trial_num, time_of_frame,
                              -1, -1, frame_num, -1]])

        def fr_paw():
            return np.array([[trial_num, time_of_frame,
                              -1, -1, -1, frame_num]])
        paw_switch = {self.blpaw: bl_paw,
                      self.brpaw: br_paw,
                      self.flpaw: fl_paw,
                      self.frpaw: fr_paw}

        df_to_append = pd.DataFrame(paw_switch[paw]().tolist(), columns=[self.trial_paw_event,
                                                                         self.time_paw_event,
                                                                         self.blpaw, self.brpaw,
                                                                         self.flpaw, self.frpaw])
        df_to_append[[self.trial_paw_event, self.blpaw, self.brpaw, self.flpaw, self.frpaw]] = \
            df_to_append[[self.trial_paw_event, self.blpaw, self.brpaw, self.flpaw, self.frpaw]].astype('int')
        paw_events = paw_events.append(df_to_append, ignore_index=True)
        paw_events.to_hdf(analysis_path+self.analysis_file_name, self.paw_events_key)

        session_hdf5.close()


    def get_all_combobox_values(self, combobox):
        return [combobox.itemText(i) for i in np.arange(combobox.count())]


    def ok_dialog(self):
        mb = QtWidgets.QMessageBox()
        mb.setText("Do you want to delete this event?")
        mb.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        answer = mb.exec()
        if answer == QtWidgets.QMessageBox.Ok:
            return True
        else:
            return False

    # Slots for mouse events (right, middle, left clicks and movement
    @QtCore.pyqtSlot(object)
    def on_mplg_frames_buttonpressed(self, event):
        button = event.button
        xdata = event.xdata
        ydata = event.ydata
        if self.data_loaded and self.analysis_exists:
            traj_name = self.ui.cB_trajectory_name.currentText() # Get name of current trajectory
            if not traj_name:  # Notify user if there is no current trajectory selected
                mb = QtWidgets.QMessageBox()
                mb.setText("You need to set a name for the trajectory.")
                mb.setStandardButtons(QtWidgets.QMessageBox.Ok)
                mb.exec()
                return
            analysis_path = os.path.join(self.session, self.analysis_folder)
            session_hdf5 = pd.HDFStore(analysis_path+self.analysis_file_name, mode='a', complevel=9, complib='zlib')
            trial_num = int(self.ui.label_trial_number_int.text())
            frame_num = int(self.ui.sBox_FrameNum.text())
            time_of_frame = session_hdf5[self.fronttime_key].iloc[frame_num]
            image_size_factor = self.ui.dSB_frame_resize.value()

            if button == 1:  # On LMB add a point to the current trial's current trajectory
                df_to_append = pd.DataFrame([[traj_name, trial_num, frame_num, time_of_frame,
                                              int(xdata/image_size_factor), int(ydata/image_size_factor)]],
                                            columns=[self.name_traj_point, self.trial_traj_point,
                                                     self.frame_traj_point, self.time_traj_point,
                                                     self.x_traj_point, self.y_traj_point])
                if self.trajectories_key in session_hdf5:
                    trajectories = session_hdf5[self.trajectories_key]; """:type : pd.DataFrame"""
                    trajectories = trajectories.append(df_to_append, ignore_index=True)
                else:
                    trajectories = df_to_append
                trajectories[self.name_traj_point] = trajectories[self.name_traj_point].astype('str')
                trajectories[[self.trial_traj_point, self.frame_traj_point, self.x_traj_point, self.y_traj_point]] = trajectories[[self.trial_traj_point, self.frame_traj_point, self.x_traj_point, self.y_traj_point]].astype('int')
                session_hdf5[self.trajectories_key] = trajectories
                self.plot_trajectory(trajectories, [traj_name], frame_num, trial_num)

            if button == 3 and self.trajectories_key in session_hdf5:  # On RMB just draws the current trial's current trajectory
                trajectories = session_hdf5[self.trajectories_key]; """:type : pd.DataFrame"""
                trials = trajectories[self.trial_traj_point]
                current_trial = int(self.ui.label_trial_number_int.text())
                if trials[trials==current_trial].size>0:  # Show only if there is a trajectory for this trial
                    self.plot_trajectory(trajectories, [traj_name], frame_num, trial_num)

            if button == 2 and self.trajectories_key in session_hdf5:  # On MMB deletes the clicked point of the current trial's current trajectory
                trajectories = session_hdf5[self.trajectories_key]; """:type : pd.DataFrame"""
                x_data = trajectories[trajectories[self.name_traj_point] == traj_name][trajectories[self.trial_traj_point] == trial_num][self.x_traj_point]
                y_data = trajectories[trajectories[self.name_traj_point] == traj_name][trajectories[self.trial_traj_point] == trial_num][self.y_traj_point]
                xdiff = np.array(x_data - xdata/image_size_factor)
                ydiff = np.array(y_data - ydata/image_size_factor)
                max_diff = 20*image_size_factor
                x_idx = np.where((xdiff>-max_diff)*(xdiff<max_diff))
                y_idx = np.where((ydiff>-max_diff)*(ydiff<max_diff))
                idx_to_remove = np.intersect1d(x_idx, y_idx)
                if x_data[idx_to_remove].size > 0 and y_data[idx_to_remove].size > 0:
                    for i in idx_to_remove:
                        x_to_remove = x_data.iloc[i].tolist()
                        y_to_remove = y_data.iloc[i].tolist()
                        traj_indices_to_remove = trajectories[trajectories[self.name_traj_point] == traj_name][trajectories[self.trial_traj_point] == trial_num][x_data == x_to_remove][y_data == y_to_remove].index
                        trajectories = trajectories.drop(traj_indices_to_remove)
                    trajectories = trajectories.reset_index(drop=True)
                    trajectories[self.name_traj_point] = trajectories[self.name_traj_point].astype('str')
                    trajectories[[self.trial_traj_point, self.frame_traj_point, self.x_traj_point, self.y_traj_point]] = trajectories[[self.trial_traj_point, self.frame_traj_point, self.x_traj_point, self.y_traj_point]].astype('int')
                    session_hdf5[self.trajectories_key] = trajectories
                self.plot_trajectory(trajectories, traj_name, frame_num, trial_num)

            session_hdf5.close()

    def plot_trajectory(self, trajectories, traj_names, frame_num, trial_num):
        self.on_sBox_FrameNum_valueChanged(frame_num)
        image_size_factor = self.ui.dSB_frame_resize.value()
        axes = self.ui.mplg_frames.all_sp_axes[0];""":type : matplotlib.axes.Axes"""
        trajectories = trajectories.sort(self.frame_traj_point, ascending=True)
        for traj_name in traj_names:
            x_data = trajectories[trajectories[self.name_traj_point] == traj_name][trajectories[self.trial_traj_point] == trial_num][self.x_traj_point]*image_size_factor
            y_data = trajectories[trajectories[self.name_traj_point] == traj_name][trajectories[self.trial_traj_point] == trial_num][self.y_traj_point]*image_size_factor

            index_of_trajectory = self.ui.cB_trajectory_name.currentIndex()
            index_to_color = {0: 'blue',
                              1: 'green',
                              2: 'yellow',
                              3: 'red'}
            color = index_to_color[index_of_trajectory % 4]
            line = axes.plot(x_data, y_data, color=color, marker='o', markersize=5)
            self.trajectory_lines.append(line[0])
        self.ui.mplg_frames.canvas.draw()

    @QtCore.pyqtSlot(object)
    def on_mplg_frames_keypressed(self, event):
        key = event.key
        if key == 'q':
            self.ui.sBox_FrameNum.setValue(self.ui.sBox_FrameNum.value()-self.ui.sBox_frames_step.value())
        if key == 'w':
            self.ui.sBox_FrameNum.setValue(self.ui.sBox_FrameNum.value()+self.ui.sBox_frames_step.value())

    @QtCore.pyqtSlot(object)
    def on_mplg_frames_mousemove(self, event):
        if self.trajectory_lines:
            for line in self.trajectory_lines:
                line.set_pickradius(5)
                if line.contains(event)[0]:
                    axes = self.ui.mplg_frames.all_sp_axes[0];""":type : matplotlib.axes.Axes"""
                    xy = line.get_xydata()
                    point_ind = line.contains(event)[1]['ind'][0]

                    traj_name = self.ui.cB_trajectory_name.currentText()
                    trial_num = int(self.ui.label_trial_number_int.text())
                    analysis_path = os.path.join(self.session, self.analysis_folder)
                    session_hdf5 = pd.HDFStore(analysis_path+self.analysis_file_name, mode='r')
                    trajectories = session_hdf5[self.trajectories_key]; """:type : pd.DataFrame"""
                    session_hdf5.close()
                    trajectories = trajectories.sort(self.frame_traj_point, ascending=True)
                    frame = trajectories[trajectories[self.name_traj_point] == traj_name][trajectories[self.trial_traj_point] == trial_num][self.frame_traj_point].tolist()[point_ind]
                    self.annotation = mpt.Annotation(str(frame), xy=tuple(xy[point_ind]), xytext=(xy[point_ind][0], xy[point_ind][1]-40), xycoords='data', textcoords='data', horizontalalignment="left",
                                           arrowprops=dict(arrowstyle="simple", connectionstyle="arc3,rad=-0.2"),
                                           bbox=dict(boxstyle="round", facecolor="w", edgecolor="0.5", alpha=0.9))
                    axes.add_artist(self.annotation)
                    self.ui.mplg_frames.canvas.draw()

    def connect_slots(self):
        self.ui.pB_selecet_data.clicked.connect(self.on_pB_select_data_clicked)
        self.ui.sBox_FrameNum.valueChanged.connect(self.on_sBox_FrameNum_valueChanged)
        self.ui.hSB_piezo_samples_to_plot.valueChanged.connect(self.on_hSB_piezo_samples_to_plot_valueChanged)
        self.ui.sBox_frames_step.valueChanged.connect(self.on_sBox_frames_step_valueChanged)
        self.ui.pB_Run.toggled.connect(self.on_pB_Run_toggled)
        self.ui.dSB_frame_resize.valueChanged.connect(self.on_dSB_frame_resize_valueChanged)
        self.ui.pB_goto_trial_frame.clicked.connect(self.on_pB_goto_trial_frame_clicked)
        self.ui.pB_add_trial.clicked.connect(self.on_pB_add_trial_clicked)
        self.ui.pB_remove_trial.clicked.connect(self.on_pB_remove_trial_clicked)
        self.ui.pB_fl_paw_add_frame.clicked.connect(self.on_pB_fl_paw_add_frame_clicked)
        self.ui.pB_fl_paw_remove_frame.clicked.connect(self.on_pB_fl_paw_remove_frame_clicked)
        self.ui.pB_fr_paw_add_frame.clicked.connect(self.on_pB_fr_paw_add_frame_clicked)
        self.ui.pB_fr_paw_remove_frame.clicked.connect(self.on_pB_fr_paw_remove_frame_clicked)
        self.ui.pB_bl_paw_add_frame.clicked.connect(self.on_pB_bl_paw_add_frame_clicked)
        self.ui.pB_bl_paw_remove_frame.clicked.connect(self.on_pB_bl_paw_remove_frame_clicked)
        self.ui.pB_br_paw_add_frame.clicked.connect(self.on_pB_br_paw_add_frame_clicked)
        self.ui.pB_br_paw_remove_frame.clicked.connect(self.on_pB_br_paw_remove_frame_clicked)
        self.ui.mplg_frames.button_pressed.connect(self.on_mplg_frames_buttonpressed)
        self.ui.mplg_frames.key_pressed.connect(self.on_mplg_frames_keypressed)
        self.ui.mplg_frames.mouse_move.connect(self.on_mplg_frames_mousemove)



RatShuttlingPawEventsGenerator()