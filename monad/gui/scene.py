"""
Monad Studio - Enhanced Scene System
Node canvas with grid background and improved wire rendering
"""

import math
from PySide6.QtWidgets import QGraphicsScene, QGraphicsPathItem
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPen, QColor, QPainterPath, QBrush, QLinearGradient, QPainter

from .nodes import BlockNode, PortItem
from .styles import Colors


class Wire(QGraphicsPathItem):
    """Bezier curve connecting two ports with gradient effect."""
    
    def __init__(self, start_port, end_port=None):
        super().__init__()
        self.start_port = start_port
        self.end_port = end_port
        
        # Default pen (will be updated in paint)
        self.setPen(QPen(QColor(Colors.WIRE_DEFAULT), 3))
        self.setZValue(-1)  # Behind nodes
        
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
        
        # Smoother Bezier curve with dynamic control points
        dx = abs(end.x() - start.x())
        dy = abs(end.y() - start.y())
        
        # Control point distance based on node separation
        ctrl_dist = max(50, min(dx * 0.5, 150))
        
        ctrl1 = QPointF(start.x() + ctrl_dist, start.y())
        ctrl2 = QPointF(end.x() - ctrl_dist, end.y())
        
        path.cubicTo(ctrl1, ctrl2, end)
        self.setPath(path)
    
    def paint(self, painter, option, widget):
        """Custom paint with gradient wire."""
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.end_port:
            # Connected wire - gradient from output to input
            start = self.start_port.scenePos()
            end = self.end_port.scenePos()
            
            gradient = QLinearGradient(start, end)
            gradient.setColorAt(0, QColor(Colors.PORT_OUTPUT))
            gradient.setColorAt(1, QColor(Colors.PORT_INPUT))
            
            pen = QPen(QBrush(gradient), 3)
            pen.setCapStyle(Qt.RoundCap)
        else:
            # Dragging wire - yellow
            pen = QPen(QColor(Colors.WIRE_DEFAULT), 3, Qt.DashLine)
            pen.setCapStyle(Qt.RoundCap)
        
        painter.strokePath(self.path(), pen)


class NodeScene(QGraphicsScene):
    """Manages the graph state with grid background."""
    
    GRID_SIZE = 25
    GRID_COLOR_LIGHT = QColor(Colors.BORDER)
    GRID_COLOR_DARK = QColor(Colors.BG_SECONDARY)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSceneRect(0, 0, 3000, 3000)
        self.setBackgroundBrush(QColor(Colors.BG_PRIMARY))
        
        self.active_wire = None
        self.nodes = []
        self.draw_grid = True

    def add_node(self, node):
        self.addItem(node)
        self.nodes.append(node)
    
    def remove_node(self, node):
        """Remove a node and its connections."""
        # Remove connected wires
        for port in node.inputs + node.outputs:
            for wire in port.wires[:]:  # Copy list to avoid modification during iteration
                self.remove_wire(wire)
        
        self.nodes.remove(node)
        self.removeItem(node)
    
    def remove_wire(self, wire):
        """Remove a wire and clean up port references."""
        if wire.start_port and wire in wire.start_port.wires:
            wire.start_port.wires.remove(wire)
        if wire.end_port and wire in wire.end_port.wires:
            wire.end_port.wires.remove(wire)
        self.removeItem(wire)

    def drawBackground(self, painter, rect):
        """Draw grid background."""
        super().drawBackground(painter, rect)
        
        if not self.draw_grid:
            return
        
        painter.setRenderHint(QPainter.Antialiasing, False)
        
        # Calculate grid bounds
        left = int(rect.left()) - (int(rect.left()) % self.GRID_SIZE)
        top = int(rect.top()) - (int(rect.top()) % self.GRID_SIZE)
        
        # Minor grid lines
        minor_pen = QPen(self.GRID_COLOR_LIGHT)
        minor_pen.setWidth(1)
        minor_pen.setStyle(Qt.DotLine)
        painter.setPen(minor_pen)
        
        # Vertical lines
        x = left
        while x < rect.right():
            if x % (self.GRID_SIZE * 4) != 0:  # Skip major lines
                painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            x += self.GRID_SIZE
        
        # Horizontal lines
        y = top
        while y < rect.bottom():
            if y % (self.GRID_SIZE * 4) != 0:  # Skip major lines
                painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
            y += self.GRID_SIZE
        
        # Major grid lines (every 4 cells)
        major_size = self.GRID_SIZE * 4
        major_left = int(rect.left()) - (int(rect.left()) % major_size)
        major_top = int(rect.top()) - (int(rect.top()) % major_size)
        
        major_pen = QPen(self.GRID_COLOR_DARK)
        major_pen.setWidth(1)
        painter.setPen(major_pen)
        
        x = major_left
        while x < rect.right():
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            x += major_size
        
        y = major_top
        while y < rect.bottom():
            painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
            y += major_size

    def mousePressEvent(self, event):
        item = self.itemAt(event.scenePos(), self.views()[0].transform())
        
        if isinstance(item, PortItem):
            if item.is_output:
                # Start wiring from output
                self.active_wire = Wire(item)
                self.addItem(self.active_wire)
            elif not item.is_output and event.button() == Qt.LeftButton:
                # Click on input - allow starting wire in reverse (for disconnecting)
                pass
        
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.active_wire:
            self.active_wire.update_path(event.scenePos())
            
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.active_wire:
            item = self.itemAt(event.scenePos(), self.views()[0].transform())
            if isinstance(item, PortItem) and not item.is_output:
                # Validate connection (prevent self-connection)
                if item.parentItem() != self.active_wire.start_port.parentItem():
                    # Connect
                    self.active_wire.end_port = item
                    self.active_wire.update_path()
                    item.wires.append(self.active_wire)
                    self.active_wire.start_port.wires.append(self.active_wire)
                    self.active_wire = None
                else:
                    # Self-connection - drop
                    self.removeItem(self.active_wire)
                    self.active_wire = None
            else:
                # Drop
                self.removeItem(self.active_wire)
                self.active_wire = None
                
        super().mouseReleaseEvent(event)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            # Delete selected nodes
            for item in self.selectedItems():
                if isinstance(item, BlockNode):
                    self.remove_node(item)
        elif event.key() == Qt.Key_G:
            # Toggle grid
            self.draw_grid = not self.draw_grid
            self.update()
        
        super().keyPressEvent(event)
