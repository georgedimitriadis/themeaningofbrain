# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'RatShuttlingPawEventsGenerator.ui'
#
# Created: Wed Mar 25 21:22:18 2015
#      by: PyQt4 UI code generator 4.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_RatShuttlingPawEventsGenerator(object):
    def setupUi(self, RatShuttlingPawEventsGenerator):
        RatShuttlingPawEventsGenerator.setObjectName(_fromUtf8("RatShuttlingPawEventsGenerator"))
        RatShuttlingPawEventsGenerator.resize(1290, 997)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(RatShuttlingPawEventsGenerator.sizePolicy().hasHeightForWidth())
        RatShuttlingPawEventsGenerator.setSizePolicy(sizePolicy)
        self.centralwidget = QtGui.QWidget(RatShuttlingPawEventsGenerator)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.mplg_frames = MplGraphWidget(self.centralwidget)
        self.mplg_frames.setGeometry(QtCore.QRect(0, -10, 1291, 591))
        self.mplg_frames.setNavBarOn(False)
        self.mplg_frames.setObjectName(_fromUtf8("mplg_frames"))
        self.mplg_piezos = MplGraphWidget(self.centralwidget)
        self.mplg_piezos.setGeometry(QtCore.QRect(0, 550, 1291, 161))
        self.mplg_piezos.setProperty("spCols", 8)
        self.mplg_piezos.setNavBarOn(False)
        self.mplg_piezos.setObjectName(_fromUtf8("mplg_piezos"))
        self.frame = QtGui.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(192, 746, 263, 139))
        self.frame.setFrameShape(QtGui.QFrame.WinPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Plain)
        self.frame.setLineWidth(4)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.layoutWidget = QtGui.QWidget(self.frame)
        self.layoutWidget.setGeometry(QtCore.QRect(4, 4, 253, 130))
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setMargin(0)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.horizontalLayout_10 = QtGui.QHBoxLayout()
        self.horizontalLayout_10.setObjectName(_fromUtf8("horizontalLayout_10"))
        self.pB_Run = QtGui.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pB_Run.setFont(font)
        self.pB_Run.setAutoFillBackground(False)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/playbuttons/Run.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/playbuttons/Stop.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.pB_Run.setIcon(icon)
        self.pB_Run.setIconSize(QtCore.QSize(64, 64))
        self.pB_Run.setCheckable(True)
        self.pB_Run.setAutoExclusive(False)
        self.pB_Run.setDefault(False)
        self.pB_Run.setFlat(True)
        self.pB_Run.setObjectName(_fromUtf8("pB_Run"))
        self.buttonGroup = QtGui.QButtonGroup(RatShuttlingPawEventsGenerator)
        self.buttonGroup.setObjectName(_fromUtf8("buttonGroup"))
        self.buttonGroup.addButton(self.pB_Run)
        self.horizontalLayout_10.addWidget(self.pB_Run)
        self.pB_Stop = QtGui.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pB_Stop.setFont(font)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(_fromUtf8(":/playbuttons/Pause.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon1.addPixmap(QtGui.QPixmap(_fromUtf8(":/playbuttons/Stop.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.pB_Stop.setIcon(icon1)
        self.pB_Stop.setIconSize(QtCore.QSize(64, 64))
        self.pB_Stop.setCheckable(True)
        self.pB_Stop.setChecked(True)
        self.pB_Stop.setAutoExclusive(False)
        self.pB_Stop.setDefault(False)
        self.pB_Stop.setFlat(True)
        self.pB_Stop.setObjectName(_fromUtf8("pB_Stop"))
        self.buttonGroup.addButton(self.pB_Stop)
        self.horizontalLayout_10.addWidget(self.pB_Stop)
        self.verticalLayout_2.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_7 = QtGui.QHBoxLayout()
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        self.label_updates_sec = QtGui.QLabel(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_updates_sec.sizePolicy().hasHeightForWidth())
        self.label_updates_sec.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_updates_sec.setFont(font)
        self.label_updates_sec.setObjectName(_fromUtf8("label_updates_sec"))
        self.horizontalLayout_7.addWidget(self.label_updates_sec)
        self.sB_updates_sec = QtGui.QSpinBox(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.sB_updates_sec.setFont(font)
        self.sB_updates_sec.setMinimum(1)
        self.sB_updates_sec.setMaximum(20)
        self.sB_updates_sec.setObjectName(_fromUtf8("sB_updates_sec"))
        self.horizontalLayout_7.addWidget(self.sB_updates_sec)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        self.frame_2 = QtGui.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(6, 746, 139, 125))
        self.frame_2.setFrameShape(QtGui.QFrame.WinPanel)
        self.frame_2.setFrameShadow(QtGui.QFrame.Plain)
        self.frame_2.setLineWidth(4)
        self.frame_2.setObjectName(_fromUtf8("frame_2"))
        self.layoutWidget1 = QtGui.QWidget(self.frame_2)
        self.layoutWidget1.setGeometry(QtCore.QRect(8, 8, 126, 112))
        self.layoutWidget1.setObjectName(_fromUtf8("layoutWidget1"))
        self.verticalLayout_8 = QtGui.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_8.setMargin(0)
        self.verticalLayout_8.setObjectName(_fromUtf8("verticalLayout_8"))
        self.pB_selecet_data = QtGui.QPushButton(self.layoutWidget1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pB_selecet_data.setFont(font)
        self.pB_selecet_data.setObjectName(_fromUtf8("pB_selecet_data"))
        self.verticalLayout_8.addWidget(self.pB_selecet_data)
        self.horizontalLayout_9 = QtGui.QHBoxLayout()
        self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
        self.label_data_loaded = QtGui.QLabel(self.layoutWidget1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_data_loaded.setFont(font)
        self.label_data_loaded.setAlignment(QtCore.Qt.AlignCenter)
        self.label_data_loaded.setObjectName(_fromUtf8("label_data_loaded"))
        self.horizontalLayout_9.addWidget(self.label_data_loaded)
        self.qled_data_loaded = QLed(self.layoutWidget1)
        self.qled_data_loaded.setProperty("value", False)
        self.qled_data_loaded.setOnColour(2)
        self.qled_data_loaded.setOffColour(1)
        self.qled_data_loaded.setShape(1)
        self.qled_data_loaded.setObjectName(_fromUtf8("qled_data_loaded"))
        self.horizontalLayout_9.addWidget(self.qled_data_loaded)
        self.verticalLayout_8.addLayout(self.horizontalLayout_9)
        self.frame_3 = QtGui.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(2, 892, 453, 75))
        self.frame_3.setFrameShape(QtGui.QFrame.WinPanel)
        self.frame_3.setFrameShadow(QtGui.QFrame.Plain)
        self.frame_3.setLineWidth(4)
        self.frame_3.setObjectName(_fromUtf8("frame_3"))
        self.layoutWidget2 = QtGui.QWidget(self.frame_3)
        self.layoutWidget2.setGeometry(QtCore.QRect(8, 4, 438, 64))
        self.layoutWidget2.setObjectName(_fromUtf8("layoutWidget2"))
        self.verticalLayout_5 = QtGui.QVBoxLayout(self.layoutWidget2)
        self.verticalLayout_5.setMargin(0)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label_FrameNum = QtGui.QLabel(self.layoutWidget2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_FrameNum.sizePolicy().hasHeightForWidth())
        self.label_FrameNum.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_FrameNum.setFont(font)
        self.label_FrameNum.setObjectName(_fromUtf8("label_FrameNum"))
        self.horizontalLayout.addWidget(self.label_FrameNum)
        self.sBox_FrameNum = QtGui.QSpinBox(self.layoutWidget2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sBox_FrameNum.sizePolicy().hasHeightForWidth())
        self.sBox_FrameNum.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.sBox_FrameNum.setFont(font)
        self.sBox_FrameNum.setFrame(True)
        self.sBox_FrameNum.setButtonSymbols(QtGui.QAbstractSpinBox.UpDownArrows)
        self.sBox_FrameNum.setSpecialValueText(_fromUtf8(""))
        self.sBox_FrameNum.setKeyboardTracking(False)
        self.sBox_FrameNum.setMaximum(10000000)
        self.sBox_FrameNum.setObjectName(_fromUtf8("sBox_FrameNum"))
        self.horizontalLayout.addWidget(self.sBox_FrameNum)
        self.pB_PrevFrame = QtGui.QPushButton(self.layoutWidget2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pB_PrevFrame.sizePolicy().hasHeightForWidth())
        self.pB_PrevFrame.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pB_PrevFrame.setFont(font)
        self.pB_PrevFrame.setObjectName(_fromUtf8("pB_PrevFrame"))
        self.horizontalLayout.addWidget(self.pB_PrevFrame)
        self.pB_NextFrame = QtGui.QPushButton(self.layoutWidget2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pB_NextFrame.sizePolicy().hasHeightForWidth())
        self.pB_NextFrame.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pB_NextFrame.setFont(font)
        self.pB_NextFrame.setObjectName(_fromUtf8("pB_NextFrame"))
        self.horizontalLayout.addWidget(self.pB_NextFrame)
        self.verticalLayout_5.addLayout(self.horizontalLayout)
        self.horizontalLayout_11 = QtGui.QHBoxLayout()
        self.horizontalLayout_11.setObjectName(_fromUtf8("horizontalLayout_11"))
        self.label_frames_step = QtGui.QLabel(self.layoutWidget2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_frames_step.sizePolicy().hasHeightForWidth())
        self.label_frames_step.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_frames_step.setFont(font)
        self.label_frames_step.setObjectName(_fromUtf8("label_frames_step"))
        self.horizontalLayout_11.addWidget(self.label_frames_step)
        self.sBox_frames_step = QtGui.QSpinBox(self.layoutWidget2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sBox_frames_step.sizePolicy().hasHeightForWidth())
        self.sBox_frames_step.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.sBox_frames_step.setFont(font)
        self.sBox_frames_step.setKeyboardTracking(False)
        self.sBox_frames_step.setMinimum(1)
        self.sBox_frames_step.setMaximum(1000)
        self.sBox_frames_step.setObjectName(_fromUtf8("sBox_frames_step"))
        self.horizontalLayout_11.addWidget(self.sBox_frames_step)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem)
        self.verticalLayout_5.addLayout(self.horizontalLayout_11)
        self.frame_4 = QtGui.QFrame(self.centralwidget)
        self.frame_4.setGeometry(QtCore.QRect(464, 746, 119, 95))
        self.frame_4.setFrameShape(QtGui.QFrame.WinPanel)
        self.frame_4.setFrameShadow(QtGui.QFrame.Plain)
        self.frame_4.setLineWidth(4)
        self.frame_4.setObjectName(_fromUtf8("frame_4"))
        self.layoutWidget3 = QtGui.QWidget(self.frame_4)
        self.layoutWidget3.setGeometry(QtCore.QRect(6, 6, 108, 85))
        self.layoutWidget3.setObjectName(_fromUtf8("layoutWidget3"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.layoutWidget3)
        self.verticalLayout_4.setMargin(0)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.label_frame_resize = QtGui.QLabel(self.layoutWidget3)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_frame_resize.sizePolicy().hasHeightForWidth())
        self.label_frame_resize.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_frame_resize.setFont(font)
        self.label_frame_resize.setAlignment(QtCore.Qt.AlignCenter)
        self.label_frame_resize.setObjectName(_fromUtf8("label_frame_resize"))
        self.verticalLayout_4.addWidget(self.label_frame_resize)
        self.dSB_frame_resize = QtGui.QDoubleSpinBox(self.layoutWidget3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.dSB_frame_resize.setFont(font)
        self.dSB_frame_resize.setDecimals(1)
        self.dSB_frame_resize.setMinimum(0.1)
        self.dSB_frame_resize.setMaximum(2.0)
        self.dSB_frame_resize.setSingleStep(0.1)
        self.dSB_frame_resize.setProperty("value", 0.5)
        self.dSB_frame_resize.setObjectName(_fromUtf8("dSB_frame_resize"))
        self.verticalLayout_4.addWidget(self.dSB_frame_resize)
        self.frame_5 = QtGui.QFrame(self.centralwidget)
        self.frame_5.setGeometry(QtCore.QRect(608, 746, 295, 111))
        self.frame_5.setFrameShape(QtGui.QFrame.WinPanel)
        self.frame_5.setFrameShadow(QtGui.QFrame.Plain)
        self.frame_5.setLineWidth(4)
        self.frame_5.setObjectName(_fromUtf8("frame_5"))
        self.layoutWidget4 = QtGui.QWidget(self.frame_5)
        self.layoutWidget4.setGeometry(QtCore.QRect(4, 4, 199, 100))
        self.layoutWidget4.setObjectName(_fromUtf8("layoutWidget4"))
        self.verticalLayout = QtGui.QVBoxLayout(self.layoutWidget4)
        self.verticalLayout.setMargin(0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.pB_add_trial = QtGui.QPushButton(self.layoutWidget4)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pB_add_trial.sizePolicy().hasHeightForWidth())
        self.pB_add_trial.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pB_add_trial.setFont(font)
        self.pB_add_trial.setObjectName(_fromUtf8("pB_add_trial"))
        self.verticalLayout.addWidget(self.pB_add_trial)
        self.pB_remove_trial = QtGui.QPushButton(self.layoutWidget4)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pB_remove_trial.setFont(font)
        self.pB_remove_trial.setObjectName(_fromUtf8("pB_remove_trial"))
        self.verticalLayout.addWidget(self.pB_remove_trial)
        self.layoutWidget5 = QtGui.QWidget(self.frame_5)
        self.layoutWidget5.setGeometry(QtCore.QRect(206, 20, 85, 71))
        self.layoutWidget5.setObjectName(_fromUtf8("layoutWidget5"))
        self.verticalLayout_9 = QtGui.QVBoxLayout(self.layoutWidget5)
        self.verticalLayout_9.setMargin(0)
        self.verticalLayout_9.setObjectName(_fromUtf8("verticalLayout_9"))
        self.label_selected_trials = QtGui.QLabel(self.layoutWidget5)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_selected_trials.sizePolicy().hasHeightForWidth())
        self.label_selected_trials.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_selected_trials.setFont(font)
        self.label_selected_trials.setAlignment(QtCore.Qt.AlignCenter)
        self.label_selected_trials.setObjectName(_fromUtf8("label_selected_trials"))
        self.verticalLayout_9.addWidget(self.label_selected_trials)
        self.cB_selected_trials = QtGui.QComboBox(self.layoutWidget5)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cB_selected_trials.sizePolicy().hasHeightForWidth())
        self.cB_selected_trials.setSizePolicy(sizePolicy)
        self.cB_selected_trials.setMaximumSize(QtCore.QSize(200, 200))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.cB_selected_trials.setFont(font)
        self.cB_selected_trials.setObjectName(_fromUtf8("cB_selected_trials"))
        self.verticalLayout_9.addWidget(self.cB_selected_trials)
        self.frame_6 = QtGui.QFrame(self.centralwidget)
        self.frame_6.setGeometry(QtCore.QRect(464, 908, 439, 59))
        self.frame_6.setFrameShape(QtGui.QFrame.WinPanel)
        self.frame_6.setFrameShadow(QtGui.QFrame.Plain)
        self.frame_6.setLineWidth(4)
        self.frame_6.setObjectName(_fromUtf8("frame_6"))
        self.layoutWidget6 = QtGui.QWidget(self.frame_6)
        self.layoutWidget6.setGeometry(QtCore.QRect(6, 6, 427, 48))
        self.layoutWidget6.setObjectName(_fromUtf8("layoutWidget6"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.layoutWidget6)
        self.horizontalLayout_2.setMargin(0)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.pB_goto_trial_frame = QtGui.QPushButton(self.layoutWidget6)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pB_goto_trial_frame.sizePolicy().hasHeightForWidth())
        self.pB_goto_trial_frame.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pB_goto_trial_frame.setFont(font)
        self.pB_goto_trial_frame.setObjectName(_fromUtf8("pB_goto_trial_frame"))
        self.horizontalLayout_2.addWidget(self.pB_goto_trial_frame)
        self.horizontalLayout_13 = QtGui.QHBoxLayout()
        self.horizontalLayout_13.setObjectName(_fromUtf8("horizontalLayout_13"))
        self.label_trial_number_txt = QtGui.QLabel(self.layoutWidget6)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_trial_number_txt.setFont(font)
        self.label_trial_number_txt.setToolTip(_fromUtf8(""))
        self.label_trial_number_txt.setAlignment(QtCore.Qt.AlignCenter)
        self.label_trial_number_txt.setObjectName(_fromUtf8("label_trial_number_txt"))
        self.horizontalLayout_13.addWidget(self.label_trial_number_txt)
        self.label_trial_number_int = QtGui.QLabel(self.layoutWidget6)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_trial_number_int.setFont(font)
        self.label_trial_number_int.setObjectName(_fromUtf8("label_trial_number_int"))
        self.horizontalLayout_13.addWidget(self.label_trial_number_int)
        self.horizontalLayout_2.addLayout(self.horizontalLayout_13)
        self.horizontalLayout_12 = QtGui.QHBoxLayout()
        self.horizontalLayout_12.setObjectName(_fromUtf8("horizontalLayout_12"))
        self.label_trial_start_frames = QtGui.QLabel(self.layoutWidget6)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_trial_start_frames.setFont(font)
        self.label_trial_start_frames.setAlignment(QtCore.Qt.AlignCenter)
        self.label_trial_start_frames.setObjectName(_fromUtf8("label_trial_start_frames"))
        self.horizontalLayout_12.addWidget(self.label_trial_start_frames)
        self.cB_trial_start_frames = QtGui.QComboBox(self.layoutWidget6)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(50)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cB_trial_start_frames.sizePolicy().hasHeightForWidth())
        self.cB_trial_start_frames.setSizePolicy(sizePolicy)
        self.cB_trial_start_frames.setMinimumSize(QtCore.QSize(150, 0))
        self.cB_trial_start_frames.setMaximumSize(QtCore.QSize(150, 30))
        self.cB_trial_start_frames.setMaxVisibleItems(20)
        self.cB_trial_start_frames.setFrame(True)
        self.cB_trial_start_frames.setObjectName(_fromUtf8("cB_trial_start_frames"))
        self.horizontalLayout_12.addWidget(self.cB_trial_start_frames)
        self.horizontalLayout_2.addLayout(self.horizontalLayout_12)
        self.frame_7 = QtGui.QFrame(self.centralwidget)
        self.frame_7.setGeometry(QtCore.QRect(8, 696, 983, 27))
        self.frame_7.setFrameShape(QtGui.QFrame.WinPanel)
        self.frame_7.setFrameShadow(QtGui.QFrame.Plain)
        self.frame_7.setLineWidth(4)
        self.frame_7.setObjectName(_fromUtf8("frame_7"))
        self.layoutWidget7 = QtGui.QWidget(self.frame_7)
        self.layoutWidget7.setGeometry(QtCore.QRect(4, 2, 975, 24))
        self.layoutWidget7.setObjectName(_fromUtf8("layoutWidget7"))
        self.horizontalLayout_8 = QtGui.QHBoxLayout(self.layoutWidget7)
        self.horizontalLayout_8.setMargin(0)
        self.horizontalLayout_8.setObjectName(_fromUtf8("horizontalLayout_8"))
        self.label_piezo_samples_to_plot = QtGui.QLabel(self.layoutWidget7)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_piezo_samples_to_plot.setFont(font)
        self.label_piezo_samples_to_plot.setObjectName(_fromUtf8("label_piezo_samples_to_plot"))
        self.horizontalLayout_8.addWidget(self.label_piezo_samples_to_plot)
        self.hSB_piezo_samples_to_plot = QtGui.QScrollBar(self.layoutWidget7)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.hSB_piezo_samples_to_plot.sizePolicy().hasHeightForWidth())
        self.hSB_piezo_samples_to_plot.setSizePolicy(sizePolicy)
        self.hSB_piezo_samples_to_plot.setMinimum(2)
        self.hSB_piezo_samples_to_plot.setMaximum(500)
        self.hSB_piezo_samples_to_plot.setSingleStep(2)
        self.hSB_piezo_samples_to_plot.setProperty("value", 100)
        self.hSB_piezo_samples_to_plot.setOrientation(QtCore.Qt.Horizontal)
        self.hSB_piezo_samples_to_plot.setObjectName(_fromUtf8("hSB_piezo_samples_to_plot"))
        self.horizontalLayout_8.addWidget(self.hSB_piezo_samples_to_plot)
        spacerItem1 = QtGui.QSpacerItem(10, 20, QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem1)
        self.label = QtGui.QLabel(self.layoutWidget7)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_8.addWidget(self.label)
        self.cB_trajectory_name = QtGui.QComboBox(self.layoutWidget7)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(100)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cB_trajectory_name.sizePolicy().hasHeightForWidth())
        self.cB_trajectory_name.setSizePolicy(sizePolicy)
        self.cB_trajectory_name.setMinimumSize(QtCore.QSize(200, 0))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.cB_trajectory_name.setFont(font)
        self.cB_trajectory_name.setEditable(True)
        self.cB_trajectory_name.setObjectName(_fromUtf8("cB_trajectory_name"))
        self.horizontalLayout_8.addWidget(self.cB_trajectory_name)
        self.frame_8 = QtGui.QFrame(self.centralwidget)
        self.frame_8.setGeometry(QtCore.QRect(912, 746, 177, 103))
        self.frame_8.setFrameShape(QtGui.QFrame.WinPanel)
        self.frame_8.setFrameShadow(QtGui.QFrame.Plain)
        self.frame_8.setLineWidth(4)
        self.frame_8.setObjectName(_fromUtf8("frame_8"))
        self.layoutWidget8 = QtGui.QWidget(self.frame_8)
        self.layoutWidget8.setGeometry(QtCore.QRect(2, -2, 171, 102))
        self.layoutWidget8.setObjectName(_fromUtf8("layoutWidget8"))
        self.gridLayout = QtGui.QGridLayout(self.layoutWidget8)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.pB_fl_paw_add_frame = QtGui.QPushButton(self.layoutWidget8)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pB_fl_paw_add_frame.setFont(font)
        self.pB_fl_paw_add_frame.setObjectName(_fromUtf8("pB_fl_paw_add_frame"))
        self.gridLayout.addWidget(self.pB_fl_paw_add_frame, 0, 0, 1, 1)
        self.pB_fl_paw_remove_frame = QtGui.QPushButton(self.layoutWidget8)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pB_fl_paw_remove_frame.setFont(font)
        self.pB_fl_paw_remove_frame.setObjectName(_fromUtf8("pB_fl_paw_remove_frame"))
        self.gridLayout.addWidget(self.pB_fl_paw_remove_frame, 0, 1, 1, 1)
        self.cB_fl_paw_frames = QtGui.QComboBox(self.layoutWidget8)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.cB_fl_paw_frames.setFont(font)
        self.cB_fl_paw_frames.setObjectName(_fromUtf8("cB_fl_paw_frames"))
        self.gridLayout.addWidget(self.cB_fl_paw_frames, 1, 1, 1, 1)
        self.label_fl_paw_frames = QtGui.QLabel(self.layoutWidget8)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_fl_paw_frames.setFont(font)
        self.label_fl_paw_frames.setAlignment(QtCore.Qt.AlignCenter)
        self.label_fl_paw_frames.setObjectName(_fromUtf8("label_fl_paw_frames"))
        self.gridLayout.addWidget(self.label_fl_paw_frames, 1, 0, 1, 1)
        self.frame_9 = QtGui.QFrame(self.centralwidget)
        self.frame_9.setGeometry(QtCore.QRect(1102, 746, 177, 103))
        self.frame_9.setFrameShape(QtGui.QFrame.WinPanel)
        self.frame_9.setFrameShadow(QtGui.QFrame.Plain)
        self.frame_9.setLineWidth(4)
        self.frame_9.setObjectName(_fromUtf8("frame_9"))
        self.layoutWidget9 = QtGui.QWidget(self.frame_9)
        self.layoutWidget9.setGeometry(QtCore.QRect(2, -2, 171, 102))
        self.layoutWidget9.setObjectName(_fromUtf8("layoutWidget9"))
        self.gridLayout_2 = QtGui.QGridLayout(self.layoutWidget9)
        self.gridLayout_2.setMargin(0)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.pB_fr_paw_add_frame = QtGui.QPushButton(self.layoutWidget9)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pB_fr_paw_add_frame.setFont(font)
        self.pB_fr_paw_add_frame.setObjectName(_fromUtf8("pB_fr_paw_add_frame"))
        self.gridLayout_2.addWidget(self.pB_fr_paw_add_frame, 0, 0, 1, 1)
        self.pB_fr_paw_remove_frame = QtGui.QPushButton(self.layoutWidget9)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pB_fr_paw_remove_frame.setFont(font)
        self.pB_fr_paw_remove_frame.setObjectName(_fromUtf8("pB_fr_paw_remove_frame"))
        self.gridLayout_2.addWidget(self.pB_fr_paw_remove_frame, 0, 1, 1, 1)
        self.cB_fr_paw_frames = QtGui.QComboBox(self.layoutWidget9)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.cB_fr_paw_frames.setFont(font)
        self.cB_fr_paw_frames.setObjectName(_fromUtf8("cB_fr_paw_frames"))
        self.gridLayout_2.addWidget(self.cB_fr_paw_frames, 1, 1, 1, 1)
        self.label_fr_paw_frames = QtGui.QLabel(self.layoutWidget9)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_fr_paw_frames.setFont(font)
        self.label_fr_paw_frames.setAlignment(QtCore.Qt.AlignCenter)
        self.label_fr_paw_frames.setObjectName(_fromUtf8("label_fr_paw_frames"))
        self.gridLayout_2.addWidget(self.label_fr_paw_frames, 1, 0, 1, 1)
        self.frame_10 = QtGui.QFrame(self.centralwidget)
        self.frame_10.setGeometry(QtCore.QRect(912, 864, 177, 103))
        self.frame_10.setFrameShape(QtGui.QFrame.WinPanel)
        self.frame_10.setFrameShadow(QtGui.QFrame.Plain)
        self.frame_10.setLineWidth(4)
        self.frame_10.setObjectName(_fromUtf8("frame_10"))
        self.layoutWidget_2 = QtGui.QWidget(self.frame_10)
        self.layoutWidget_2.setGeometry(QtCore.QRect(2, -2, 171, 102))
        self.layoutWidget_2.setObjectName(_fromUtf8("layoutWidget_2"))
        self.gridLayout_3 = QtGui.QGridLayout(self.layoutWidget_2)
        self.gridLayout_3.setMargin(0)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.pB_bl_paw_add_frame = QtGui.QPushButton(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pB_bl_paw_add_frame.setFont(font)
        self.pB_bl_paw_add_frame.setObjectName(_fromUtf8("pB_bl_paw_add_frame"))
        self.gridLayout_3.addWidget(self.pB_bl_paw_add_frame, 0, 0, 1, 1)
        self.pB_bl_paw_remove_frame = QtGui.QPushButton(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pB_bl_paw_remove_frame.setFont(font)
        self.pB_bl_paw_remove_frame.setObjectName(_fromUtf8("pB_bl_paw_remove_frame"))
        self.gridLayout_3.addWidget(self.pB_bl_paw_remove_frame, 0, 1, 1, 1)
        self.cB_bl_paw_frames = QtGui.QComboBox(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.cB_bl_paw_frames.setFont(font)
        self.cB_bl_paw_frames.setObjectName(_fromUtf8("cB_bl_paw_frames"))
        self.gridLayout_3.addWidget(self.cB_bl_paw_frames, 1, 1, 1, 1)
        self.label_bl_paw_frames = QtGui.QLabel(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_bl_paw_frames.setFont(font)
        self.label_bl_paw_frames.setAlignment(QtCore.Qt.AlignCenter)
        self.label_bl_paw_frames.setObjectName(_fromUtf8("label_bl_paw_frames"))
        self.gridLayout_3.addWidget(self.label_bl_paw_frames, 1, 0, 1, 1)
        self.frame_11 = QtGui.QFrame(self.centralwidget)
        self.frame_11.setGeometry(QtCore.QRect(1102, 864, 177, 103))
        self.frame_11.setFrameShape(QtGui.QFrame.WinPanel)
        self.frame_11.setFrameShadow(QtGui.QFrame.Plain)
        self.frame_11.setLineWidth(4)
        self.frame_11.setObjectName(_fromUtf8("frame_11"))
        self.layoutWidget_3 = QtGui.QWidget(self.frame_11)
        self.layoutWidget_3.setGeometry(QtCore.QRect(2, -2, 171, 102))
        self.layoutWidget_3.setObjectName(_fromUtf8("layoutWidget_3"))
        self.gridLayout_4 = QtGui.QGridLayout(self.layoutWidget_3)
        self.gridLayout_4.setMargin(0)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.pB_br_paw_add_frame = QtGui.QPushButton(self.layoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pB_br_paw_add_frame.setFont(font)
        self.pB_br_paw_add_frame.setObjectName(_fromUtf8("pB_br_paw_add_frame"))
        self.gridLayout_4.addWidget(self.pB_br_paw_add_frame, 0, 0, 1, 1)
        self.pB_br_paw_remove_frame = QtGui.QPushButton(self.layoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pB_br_paw_remove_frame.setFont(font)
        self.pB_br_paw_remove_frame.setObjectName(_fromUtf8("pB_br_paw_remove_frame"))
        self.gridLayout_4.addWidget(self.pB_br_paw_remove_frame, 0, 1, 1, 1)
        self.cB_br_paw_frames = QtGui.QComboBox(self.layoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.cB_br_paw_frames.setFont(font)
        self.cB_br_paw_frames.setObjectName(_fromUtf8("cB_br_paw_frames"))
        self.gridLayout_4.addWidget(self.cB_br_paw_frames, 1, 1, 1, 1)
        self.label_br_paw_frames = QtGui.QLabel(self.layoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_br_paw_frames.setFont(font)
        self.label_br_paw_frames.setAlignment(QtCore.Qt.AlignCenter)
        self.label_br_paw_frames.setObjectName(_fromUtf8("label_br_paw_frames"))
        self.gridLayout_4.addWidget(self.label_br_paw_frames, 1, 0, 1, 1)
        self.label_paw_touch_frames = QtGui.QLabel(self.centralwidget)
        self.label_paw_touch_frames.setGeometry(QtCore.QRect(982, 720, 231, 27))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_paw_touch_frames.setFont(font)
        self.label_paw_touch_frames.setObjectName(_fromUtf8("label_paw_touch_frames"))
        self.layoutWidget10 = QtGui.QWidget(self.centralwidget)
        self.layoutWidget10.setGeometry(QtCore.QRect(0, 0, 2, 2))
        self.layoutWidget10.setObjectName(_fromUtf8("layoutWidget10"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.layoutWidget10)
        self.verticalLayout_3.setMargin(0)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        RatShuttlingPawEventsGenerator.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(RatShuttlingPawEventsGenerator)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1290, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        RatShuttlingPawEventsGenerator.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(RatShuttlingPawEventsGenerator)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        RatShuttlingPawEventsGenerator.setStatusBar(self.statusbar)

        self.retranslateUi(RatShuttlingPawEventsGenerator)
        self.cB_trial_start_frames.setCurrentIndex(-1)
        QtCore.QObject.connect(self.pB_PrevFrame, QtCore.SIGNAL(_fromUtf8("released()")), self.sBox_FrameNum.stepDown)
        QtCore.QObject.connect(self.pB_NextFrame, QtCore.SIGNAL(_fromUtf8("released()")), self.sBox_FrameNum.stepUp)
        QtCore.QMetaObject.connectSlotsByName(RatShuttlingPawEventsGenerator)

    def retranslateUi(self, RatShuttlingPawEventsGenerator):
        RatShuttlingPawEventsGenerator.setWindowTitle(_translate("RatShuttlingPawEventsGenerator", "Rat Shuttling Paw Events Generator", None))
        self.pB_Run.setText(_translate("RatShuttlingPawEventsGenerator", "Run", None))
        self.pB_Stop.setText(_translate("RatShuttlingPawEventsGenerator", "Pause", None))
        self.label_updates_sec.setText(_translate("RatShuttlingPawEventsGenerator", "Updates\n"
"per sec", None))
        self.pB_selecet_data.setText(_translate("RatShuttlingPawEventsGenerator", "Select Data\n"
"Directory", None))
        self.label_data_loaded.setText(_translate("RatShuttlingPawEventsGenerator", "No Data\n"
"Loaded", None))
        self.label_FrameNum.setText(_translate("RatShuttlingPawEventsGenerator", "Frame Number:", None))
        self.pB_PrevFrame.setText(_translate("RatShuttlingPawEventsGenerator", "Previous Frame", None))
        self.pB_NextFrame.setText(_translate("RatShuttlingPawEventsGenerator", "Next Frame", None))
        self.label_frames_step.setText(_translate("RatShuttlingPawEventsGenerator", "Frame Step Size:", None))
        self.label_frame_resize.setText(_translate("RatShuttlingPawEventsGenerator", "Frame resize\n"
"factor", None))
        self.pB_add_trial.setText(_translate("RatShuttlingPawEventsGenerator", "Add Current Trial\n"
"To Selected List", None))
        self.pB_remove_trial.setText(_translate("RatShuttlingPawEventsGenerator", "Remove Shown Trial From\n"
"Selected List", None))
        self.label_selected_trials.setText(_translate("RatShuttlingPawEventsGenerator", "Selected\n"
"Trials", None))
        self.pB_goto_trial_frame.setText(_translate("RatShuttlingPawEventsGenerator", "Go To Trial\n"
"Start Frame", None))
        self.label_trial_number_txt.setText(_translate("RatShuttlingPawEventsGenerator", "Current \n"
"Trial", None))
        self.label_trial_number_int.setText(_translate("RatShuttlingPawEventsGenerator", "-1", None))
        self.label_trial_start_frames.setText(_translate("RatShuttlingPawEventsGenerator", "Trials\n"
"Start Frames", None))
        self.label_piezo_samples_to_plot.setText(_translate("RatShuttlingPawEventsGenerator", "Number of Ploted Piezo Samples", None))
        self.label.setText(_translate("RatShuttlingPawEventsGenerator", "Current Trajectory Name", None))
        self.pB_fl_paw_add_frame.setText(_translate("RatShuttlingPawEventsGenerator", "Add Front\n"
"Left Paw", None))
        self.pB_fl_paw_remove_frame.setText(_translate("RatShuttlingPawEventsGenerator", "Remove Front\n"
"Left Paw", None))
        self.label_fl_paw_frames.setText(_translate("RatShuttlingPawEventsGenerator", "Frames of\n"
"FL Paw", None))
        self.pB_fr_paw_add_frame.setText(_translate("RatShuttlingPawEventsGenerator", "Add Front\n"
"Right Paw", None))
        self.pB_fr_paw_remove_frame.setText(_translate("RatShuttlingPawEventsGenerator", "Remove Front\n"
"Right Paw", None))
        self.label_fr_paw_frames.setText(_translate("RatShuttlingPawEventsGenerator", "Frames of\n"
"FR Paw", None))
        self.pB_bl_paw_add_frame.setText(_translate("RatShuttlingPawEventsGenerator", "Add Back\n"
"Left Paw", None))
        self.pB_bl_paw_remove_frame.setText(_translate("RatShuttlingPawEventsGenerator", "Remove Back\n"
"Left Paw", None))
        self.label_bl_paw_frames.setText(_translate("RatShuttlingPawEventsGenerator", "Frames of\n"
"BL Paw", None))
        self.pB_br_paw_add_frame.setText(_translate("RatShuttlingPawEventsGenerator", "Add Back\n"
"Right Paw", None))
        self.pB_br_paw_remove_frame.setText(_translate("RatShuttlingPawEventsGenerator", "Remove Back\n"
"Right Paw", None))
        self.label_br_paw_frames.setText(_translate("RatShuttlingPawEventsGenerator", "Frames of\n"
"BR Paw", None))
        self.label_paw_touch_frames.setText(_translate("RatShuttlingPawEventsGenerator", "Adding Paw Touch Frames", None))

from qled import QLed
from mplgraphwidget import MplGraphWidget
