

import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
from . import correct_pyqtgraph_roi as cpg
from . import custom_viewboxmenu


class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.setMouseMode(self.RectMode)
        self.pan_mode = False
        self.square_roi_on = 0
        self.freeform_roi_on = 0
        self.freeform_roi_positions = []
        self.all_rois = []
        self.connect_on_roi_select = None
        self.connect_on_roi_delete = None
        self.number_of_rois = 1
        self.menu = custom_viewboxmenu.ViewBoxMenu(self)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            if self.square_roi_on == 0 and self.freeform_roi_on == 0:
                self.autoRange()
            else:
                if self.raiseContextMenu(ev):
                    ev.accept()

        if ev.button() == QtCore.Qt.LeftButton:
            if self.freeform_roi_on > 0:
                if len(self.all_rois) < self.number_of_rois:
                    self.freeform_roi_on = 1
                    self.freeform_roi_positions = []

                position = self.mapToView(ev.pos())
                self.freeform_roi_positions.append(position)

                if self.freeform_roi_on == 1:
                    if len(self.all_rois) == self.number_of_rois:
                        self.removeItem(self.all_rois[-1])
                    self.all_rois.append(cpg.PolyLineROI(positions=self.freeform_roi_positions, closed=True, movable=False, parent=self))
                    self.all_rois[-1].sigRegionChangeFinished.connect(self.freeform_roi_handle_move)
                    self.all_rois[-1].sigClicked.connect(self.freeform_roi_update)
                    self.addItem(self.all_rois[-1])
                    self.connect_on_roi_select(self.all_rois, freeform=True)
                else:
                    self.freeform_roi_update()
                self.freeform_roi_on += 1

        ev.accept()

    def mouseDragEvent(self, ev):
        if self.square_roi_on == 0 and self.freeform_roi_on == 0:
            if ev.button() == QtCore.Qt.RightButton:
                ev.ignore()
            else:
                pg.ViewBox.mouseDragEvent(self, ev)

        elif self.square_roi_on > 0 and self.freeform_roi_on == 0:
            start = self.mapToView(ev.buttonDownPos())
            if self.square_roi_on == 1:
                if len(self.all_rois) == self.number_of_rois:
                    self.delete_square_roi()
                self.all_rois.append(pg.RectROI(start, [0, 0]))
                self.addItem(self.all_rois[-1])
                self.square_roi_on = 2

            if not ev.isStart():
                current = self.mapToView(ev.pos())
                self.all_rois[-1].setSize(current - start)

            if ev.isFinish():
                self.square_roi_on = 1
                self.connect_on_roi_select(self.all_rois, freeform=False)
                ev.ignore()
                return

            ev.accept()

    def set_to_pan_mode(self):
        self.setMouseMode(self.PanMode)
        self.pan_mode = True
        self.delete_square_roi(mode=0)
        self.delete_freeform_roi(mode=0)

    def set_to_rect_mode(self):
        self.setMouseMode(self.RectMode)
        self.pan_mode = False
        self.delete_square_roi(mode=0)
        self.delete_freeform_roi(mode=0)

    def draw_square_roi(self):
        self.square_roi_on = 1
        self.freeform_roi_on = 0
        self.pan_mode = False
        self.delete_freeform_roi(mode=0)

    def draw_freeform_roi(self):
        self.freeform_roi_on = 1
        self.square_roi_on = 0
        self.pan_mode = False
        self.delete_square_roi(mode=0)

    def delete_square_roi(self, mode=1):
        self.square_roi_on = mode
        for i in range(len(self.all_rois)):
            self.removeItem(self.all_rois[i])
        self.all_rois = []
        self.number_of_rois = 1
        self.connect_on_roi_delete()

    def delete_freeform_roi(self, mode=1):
        self.freeform_roi_on = mode
        for i in range(len(self.all_rois)):
            self.removeItem(self.all_rois[i])
        self.all_rois = []
        self.number_of_rois = 1
        self.freeform_roi_positions = []
        self.connect_on_roi_delete()

    # TO DO: Currently only moving the handles of the latest ROI will update anything.
    # Moving the handles of previous ROIs does nothing
    def freeform_roi_handle_move(self):
        positions = [name_pos[1] for name_pos in self.all_rois[-1].getLocalHandlePositions()]
        if len(positions) == len(self.freeform_roi_positions):
            same = True
            for p in range(len(positions)):
                different = 0
                for f in range(len(self.freeform_roi_positions)):
                    if positions[p] != self.freeform_roi_positions[f]:
                        different += 1
                if different == len(self.freeform_roi_positions):
                    same = False
                    break

            if not same:
                self.freeform_roi_positions = positions
                self.all_rois[-1].setPoints(positions)
                self.connect_on_roi_select(self.all_rois, freeform=True)

    def freeform_roi_update(self):
        print(self.freeform_roi_positions)
        self.all_rois[-1].setPoints(self.freeform_roi_positions, closed=True)
        self.connect_on_roi_select(self.all_rois, freeform=True)
