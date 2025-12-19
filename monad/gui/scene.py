
import math
from PySide6.QtWidgets import QGraphicsScene, QGraphicsPathItem
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPen, QColor, QPainterPath

from .nodes import BlockNode, PortItem

class Wire(QGraphicsPathItem):
    """Bezier Curve connecting two ports."""
    def __init__(self, start_port, end_port=None):
        super().__init__()
        self.start_port = start_port
        self.end_port = end_port
        
        self.setPen(QPen(QColor("#f1c40f"), 3))
        self.setZValue(-1) # Behind nodes
        
        self.update_path()
        
    def update_path(self, mouse_pos=None):
        start = self.start_port.scenePos()
        if self.end_port:
            end = self.end_port.scenePos()
        elif mouse_pos:
            end = mouse_pos
        else:
            return

        path = QPainterPath()
        path.moveTo(start)
        
        # Cubic Bezier Control Points
        dx = end.x() - start.x()
        ctrl1 = QPointF(start.x() + dx * 0.5, start.y())
        ctrl2 = QPointF(end.x() - dx * 0.5, end.y())
        
        path.cubicTo(ctrl1, ctrl2, end)
        self.setPath(path)

class NodeScene(QGraphicsScene):
    """Manages the graph state."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSceneRect(0, 0, 2000, 2000)
        self.setBackgroundBrush(QColor("#2c3e50")) # Dark background
        
        self.active_wire = None
        self.nodes = []

    def add_node(self, node):
        self.addItem(node)
        self.nodes.append(node)

    def mousePressEvent(self, event):
        item = self.itemAt(event.scenePos(), self.views()[0].transform())
        
        if isinstance(item, PortItem):
            if item.is_output:
                # Start wiring
                self.active_wire = Wire(item)
                self.addItem(self.active_wire)
        
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.active_wire:
            self.active_wire.update_path(event.scenePos())
            
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.active_wire:
            item = self.itemAt(event.scenePos(), self.views()[0].transform())
            if isinstance(item, PortItem) and not item.is_output:
                # Connect
                self.active_wire.end_port = item
                self.active_wire.update_path()
                item.wires.append(self.active_wire)
                self.active_wire.start_port.wires.append(self.active_wire)
                self.active_wire = None
            else:
                # Drop
                self.removeItem(self.active_wire)
                self.active_wire = None
                
        super().mouseReleaseEvent(event)
