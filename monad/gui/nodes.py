
from PySide6.QtWidgets import QGraphicsItem, QGraphicsTextItem, QGraphicsProxyWidget, QLabel, QSpinBox, QLayout
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QBrush, QPen, QColor, QFont, QPainter

class PortItem(QGraphicsItem):
    """A connection port (Input/Output bubble)."""
    RADIUS = 6
    
    def __init__(self, parent, name, is_output=False, port_type="scalar"):
        super().__init__(parent)
        self.name = name
        self.is_output = is_output
        self.port_type = port_type # 'scalar', 'vector', 'matrix'
        self.wires = []
        
        # Appearance
        self.brush = QBrush(QColor("#e74c3c") if is_output else QColor("#2ecc71"))
        self.pen = QPen(QColor("#2c3e50"), 1)
        
        # Label
        self.label = QGraphicsTextItem(name, self)
        self.label.setFont(QFont("Arial", 8))
        if is_output:
            self.label.setPos(-self.label.boundingRect().width() - 10, -10)
        else:
            self.label.setPos(10, -10)

    def boundingRect(self):
        return QRectF(-self.RADIUS, -self.RADIUS, 2*self.RADIUS, 2*self.RADIUS)
        
    def paint(self, painter, option, widget):
        painter.setBrush(self.brush)
        painter.setPen(self.pen)
        painter.drawEllipse(-self.RADIUS, -self.RADIUS, 2*self.RADIUS, 2*self.RADIUS)

class BlockNode(QGraphicsItem):
    """A functioning Visual Block (Households, Policy, etc)."""
    
    def __init__(self, title="Block", inputs=None, outputs=None, pos=None, node_type="generic", params=None):
        super().__init__()
        self.title = title
        self.node_type = node_type
        self.params = params if params else {}
        self.width = 160
        self.height = 100 # Dynamic later
        
        self.inputs = []
        self.outputs = []
        
        # State
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        
        if pos:
            self.setPos(pos[0], pos[1])
            
        # UI Setup
        self._setup_header()
        
        # Add Ports
        if inputs:
            for i, name in enumerate(inputs):
                self.add_port(name, is_output=False, idx=i, total=len(inputs))
        
        if outputs:
            for i, name in enumerate(outputs):
                self.add_port(name, is_output=True, idx=i, total=len(outputs))
                
    def _setup_header(self):
        self.title_item = QGraphicsTextItem(self.title, self)
        self.title_item.setFont(QFont("Arial", 10, QFont.Bold))
        self.title_item.setDefaultTextColor(QColor("white"))
        self.title_item.setPos(10, 5)

    def add_port(self, name, is_output, idx, total):
        port = PortItem(self, name, is_output)
        
        # Spacing
        y_step = (self.height - 30) / (total + 1) if total > 0 else 0
        y = 30 + (idx + 1) * y_step
        
        if is_output:
            port.setPos(self.width, y)
            self.outputs.append(port)
        else:
            port.setPos(0, y)
            self.inputs.append(port)

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)
        
    def paint(self, painter, option, widget):
        # Body
        painter.setBrush(QBrush(QColor("#34495e")))
        painter.setPen(QPen(QColor("#2c3e50"), 2))
        painter.drawRoundedRect(0, 0, self.width, self.height, 5, 5)
        
        # Header
        painter.setBrush(QBrush(QColor("#2980b9")))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, self.width, 25, 5, 5)
        painter.drawRect(0, 20, self.width, 5) # Fill bottom corners of header
