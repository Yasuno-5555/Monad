"""
Monad Studio - Enhanced Node System
Visual blocks with modern styling, gradients, and subtle effects
"""

from PySide6.QtWidgets import QGraphicsItem, QGraphicsTextItem, QGraphicsDropShadowEffect
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QBrush, QPen, QColor, QFont, QPainter, QLinearGradient, QPainterPath

from .styles import Colors, Fonts, create_header_gradient


class PortItem(QGraphicsItem):
    """A connection port (Input/Output) with modern styling."""
    RADIUS = 7
    
    def __init__(self, parent, name, is_output=False, port_type="scalar"):
        super().__init__(parent)
        self.name = name
        self.is_output = is_output
        self.port_type = port_type  # 'scalar', 'vector', 'matrix'
        self.wires = []
        self.hovered = False
        
        # Enable hover events
        self.setAcceptHoverEvents(True)
        
        # Colors based on port direction
        if is_output:
            self.color = QColor(Colors.PORT_OUTPUT)
        else:
            self.color = QColor(Colors.PORT_INPUT)
        
        self.border_color = QColor(Colors.BG_PRIMARY)
        
        # Label
        self.label = QGraphicsTextItem(name, self)
        self.label.setFont(Fonts.small())
        self.label.setDefaultTextColor(QColor(Colors.TEXT_SECONDARY))
        
        if is_output:
            self.label.setPos(-self.label.boundingRect().width() - 12, -10)
        else:
            self.label.setPos(12, -10)

    def boundingRect(self):
        r = self.RADIUS + 2
        return QRectF(-r, -r, 2*r, 2*r)
        
    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)
        
        radius = self.RADIUS + 2 if self.hovered else self.RADIUS
        
        # Outer glow on hover
        if self.hovered:
            glow_color = QColor(self.color)
            glow_color.setAlpha(80)
            painter.setBrush(QBrush(glow_color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(-radius - 3, -radius - 3, (radius + 3) * 2, (radius + 3) * 2)
        
        # Main circle with gradient
        gradient = QLinearGradient(-radius, -radius, radius, radius)
        gradient.setColorAt(0, self.color.lighter(120))
        gradient.setColorAt(1, self.color.darker(110))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(self.border_color, 2))
        painter.drawEllipse(-radius, -radius, radius * 2, radius * 2)
        
        # Inner highlight
        inner_r = radius * 0.4
        painter.setBrush(QBrush(QColor(255, 255, 255, 60)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(-inner_r, -inner_r - radius * 0.2, inner_r * 2, inner_r * 2)
    
    def hoverEnterEvent(self, event):
        self.hovered = True
        self.update()
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        self.hovered = False
        self.update()
        super().hoverLeaveEvent(event)


class BlockNode(QGraphicsItem):
    """A visual block representing economic model components with modern styling."""
    
    CORNER_RADIUS = 10
    HEADER_HEIGHT = 32
    SHADOW_BLUR = 15
    
    def __init__(self, title="Block", inputs=None, outputs=None, pos=None, 
                 node_type="generic", params=None):
        super().__init__()
        self.title = title
        self.node_type = node_type
        self.params = params if params else {}
        self.width = 180
        self.height = 110
        
        self.inputs = []
        self.outputs = []
        
        # State
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        
        self.hovered = False
        self.selected_state = False
        
        if pos:
            self.setPos(pos[0], pos[1])
        
        # UI Setup
        self._setup_header()
        
        # Calculate height based on ports
        num_ports = max(len(inputs) if inputs else 0, len(outputs) if outputs else 0)
        self.height = max(110, self.HEADER_HEIGHT + 20 + num_ports * 28)
        
        # Add Ports
        if inputs:
            for i, name in enumerate(inputs):
                self.add_port(name, is_output=False, idx=i, total=len(inputs))
        
        if outputs:
            for i, name in enumerate(outputs):
                self.add_port(name, is_output=True, idx=i, total=len(outputs))
    
    def _setup_header(self):
        # Title
        self.title_item = QGraphicsTextItem(self.title, self)
        self.title_item.setFont(Fonts.subtitle())
        self.title_item.setDefaultTextColor(QColor(Colors.TEXT_PRIMARY))
        self.title_item.setPos(12, 6)
        
        # Type indicator icon
        icon_map = {
            "household": "👥",
            "policy": "🏛",
            "market": "📊",
            "fiscal": "💰",
        }
        icon = icon_map.get(self.node_type, "◆")
        self.icon_item = QGraphicsTextItem(icon, self)
        self.icon_item.setFont(QFont("Segoe UI Emoji", 10))
        self.icon_item.setPos(self.width - 28, 5)

    def add_port(self, name, is_output, idx, total):
        port = PortItem(self, name, is_output)
        
        # Spacing
        y_start = self.HEADER_HEIGHT + 15
        y_step = (self.height - y_start - 15) / (total) if total > 0 else 0
        y = y_start + idx * y_step + y_step / 2
        
        if is_output:
            port.setPos(self.width, y)
            self.outputs.append(port)
        else:
            port.setPos(0, y)
            self.inputs.append(port)

    def boundingRect(self):
        # Include shadow area
        margin = self.SHADOW_BLUR
        return QRectF(-margin, -margin, self.width + margin * 2, self.height + margin * 2)
    
    def shape(self):
        """Define the clickable area (excluding shadow)."""
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width, self.height, 
                           self.CORNER_RADIUS, self.CORNER_RADIUS)
        return path
        
    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)
        
        is_selected = self.isSelected()
        
        # Draw shadow (simple offset, performance-friendly)
        if not is_selected:
            shadow_offset = 4
            shadow_color = QColor(0, 0, 0, 50)
            painter.setBrush(QBrush(shadow_color))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(shadow_offset, shadow_offset, 
                                   self.width, self.height,
                                   self.CORNER_RADIUS, self.CORNER_RADIUS)
        
        # Selection glow effect
        if is_selected:
            glow_color = QColor(Colors.HIGHLIGHT)
            glow_color.setAlpha(100)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(glow_color, 4))
            painter.drawRoundedRect(-2, -2, self.width + 4, self.height + 4,
                                   self.CORNER_RADIUS + 2, self.CORNER_RADIUS + 2)
        
        # Body background with subtle gradient
        body_gradient = QLinearGradient(0, 0, 0, self.height)
        body_gradient.setColorAt(0, QColor(Colors.BG_CARD))
        body_gradient.setColorAt(1, QColor(Colors.BG_SECONDARY))
        
        painter.setBrush(QBrush(body_gradient))
        
        # Border color based on selection
        if is_selected:
            painter.setPen(QPen(QColor(Colors.HIGHLIGHT), 2))
        elif self.hovered:
            painter.setPen(QPen(QColor(Colors.BORDER_FOCUS), 2))
        else:
            painter.setPen(QPen(QColor(Colors.BORDER), 1))
        
        painter.drawRoundedRect(0, 0, self.width, self.height, 
                               self.CORNER_RADIUS, self.CORNER_RADIUS)
        
        # Header with gradient
        header_path = QPainterPath()
        header_path.addRoundedRect(0, 0, self.width, self.HEADER_HEIGHT, 
                                   self.CORNER_RADIUS, self.CORNER_RADIUS)
        # Cut off bottom rounded corners of header
        header_path.addRect(0, self.HEADER_HEIGHT - self.CORNER_RADIUS, 
                           self.width, self.CORNER_RADIUS)
        
        header_gradient = create_header_gradient(self.node_type, self.width)
        painter.setBrush(QBrush(header_gradient))
        painter.setPen(Qt.NoPen)
        
        # Clip to header area
        clip_path = QPainterPath()
        clip_path.addRoundedRect(0, 0, self.width, self.height,
                                self.CORNER_RADIUS, self.CORNER_RADIUS)
        clip_path2 = QPainterPath()
        clip_path2.addRect(0, 0, self.width, self.HEADER_HEIGHT)
        
        painter.setClipPath(clip_path)
        painter.drawRect(0, 0, self.width, self.HEADER_HEIGHT)
        painter.setClipping(False)
        
        # Header bottom line
        painter.setPen(QPen(QColor(0, 0, 0, 30), 1))
        painter.drawLine(0, self.HEADER_HEIGHT, self.width, self.HEADER_HEIGHT)
    
    def hoverEnterEvent(self, event):
        self.hovered = True
        self.update()
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        self.hovered = False
        self.update()
        super().hoverLeaveEvent(event)
    
    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            # Update connected wires when node moves
            for port in self.inputs + self.outputs:
                for wire in port.wires:
                    wire.update_path()
        return super().itemChange(change, value)
