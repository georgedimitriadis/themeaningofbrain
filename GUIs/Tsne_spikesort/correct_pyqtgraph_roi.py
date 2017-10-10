

import pyqtgraph as pg
from pyqtgraph import QtCore
from pyqtgraph import functions as fn


class PolyLineROI(pg.PolyLineROI):

    def __init__(self, positions, closed=False, pos=None, **args):
        pg.PolyLineROI.__init__(self, positions, closed=False, pos=None, **args)

    def addSegment(self, h1, h2, index=None):
        seg = _PolyLineSegment(handles=(h1, h2), pen=self.pen, parent=self, movable=False)
        if index is None:
            self.segments.append(seg)
        else:
            self.segments.insert(index, seg)
        seg.sigClicked.connect(self.segmentClicked)
        seg.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        seg.setZValue(self.zValue() + 1)
        for h in seg.handles:
            h['item'].setDeletable(True)
            h['item'].setAcceptedMouseButtons(h['item'].acceptedMouseButtons() | QtCore.Qt.LeftButton)  ## have these handles take left clicks too, so that handles cannot be added on top of other handles


class _PolyLineSegment(pg.LineSegmentROI):
    # Used internally by PolyLineROI
    def __init__(self, *args, **kwds):
        self._parentHovering = False
        pg.LineSegmentROI.__init__(self, *args, **kwds)

    def setParentHover(self, hover):
        # set independently of own hover state
        if self._parentHovering != hover:
            self._parentHovering = hover
            self._updateHoverColor()

    def _makePen(self):
        if self.mouseHovering or self._parentHovering:
            return fn.mkPen(255, 255, 0)
        else:
            return self.pen

    def hoverEvent(self, ev):
        # accept drags even though we discard them to prevent competition with parent ROI
        # (unless parent ROI is not movable)
        if self.parentItem() is not None and self.parentItem().translatable:
            ev.acceptDrags(QtCore.Qt.LeftButton)
        return pg.LineSegmentROI.hoverEvent(self, ev)