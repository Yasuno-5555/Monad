"""
Monad Studio - Centralized Style System
Modern dark theme with performance-conscious design
"""

from PySide6.QtGui import QColor, QLinearGradient, QBrush, QPen, QFont
from PySide6.QtCore import Qt

# =============================================================================
# COLOR PALETTE
# =============================================================================

class Colors:
    """Centralized color definitions for consistent theming."""
    
    # Background Colors
    BG_PRIMARY = "#1a1a2e"
    BG_SECONDARY = "#16213e"
    BG_TERTIARY = "#0f3460"
    BG_CARD = "#1e1e3f"
    
    # Accent Colors
    ACCENT = "#0f3460"
    HIGHLIGHT = "#e94560"
    SUCCESS = "#00d9a9"
    WARNING = "#f59e0b"
    ERROR = "#ef4444"
    
    # Text Colors
    TEXT_PRIMARY = "#eaeaea"
    TEXT_SECONDARY = "#a0a0a0"
    TEXT_MUTED = "#6b7280"
    
    # Border Colors
    BORDER = "#3d3d5c"
    BORDER_FOCUS = "#6366f1"
    
    # Node Type Colors (Base → Gradient End)
    NODE_HOUSEHOLD = ("#6366f1", "#8b5cf6")  # Indigo → Purple
    NODE_POLICY = ("#f59e0b", "#f97316")      # Amber → Orange
    NODE_MARKET = ("#10b981", "#14b8a6")      # Emerald → Teal
    NODE_FISCAL = ("#ec4899", "#f43f5e")      # Pink → Rose
    NODE_GENERIC = ("#64748b", "#475569")     # Slate
    
    # Wire Colors
    WIRE_DEFAULT = "#f1c40f"
    WIRE_ACTIVE = "#00d9a9"
    WIRE_INVALID = "#ef4444"
    
    # Port Colors
    PORT_INPUT = "#00d9a9"
    PORT_OUTPUT = "#e94560"
    
    @classmethod
    def get_node_colors(cls, node_type: str) -> tuple:
        """Get gradient colors for a node type."""
        color_map = {
            "household": cls.NODE_HOUSEHOLD,
            "policy": cls.NODE_POLICY,
            "market": cls.NODE_MARKET,
            "fiscal": cls.NODE_FISCAL,
        }
        return color_map.get(node_type, cls.NODE_GENERIC)


# =============================================================================
# FONTS
# =============================================================================

class Fonts:
    """Font definitions for consistent typography."""
    
    @staticmethod
    def title() -> QFont:
        font = QFont("Segoe UI", 11)
        font.setBold(True)
        return font
    
    @staticmethod
    def subtitle() -> QFont:
        font = QFont("Segoe UI", 9)
        font.setBold(True)
        return font
    
    @staticmethod
    def body() -> QFont:
        return QFont("Segoe UI", 9)
    
    @staticmethod
    def small() -> QFont:
        return QFont("Segoe UI", 8)
    
    @staticmethod
    def mono() -> QFont:
        return QFont("Consolas", 9)


# =============================================================================
# QSS STYLESHEETS
# =============================================================================

def get_main_stylesheet() -> str:
    """Return the main application stylesheet."""
    return f"""
    /* Main Window */
    QMainWindow {{
        background-color: {Colors.BG_PRIMARY};
    }}
    
    /* Toolbar */
    QToolBar {{
        background-color: {Colors.BG_SECONDARY};
        border: none;
        padding: 8px;
        spacing: 6px;
    }}
    
    QToolBar::separator {{
        background-color: {Colors.BORDER};
        width: 1px;
        margin: 4px 8px;
    }}
    
    /* Push Buttons */
    QPushButton {{
        background-color: {Colors.BG_TERTIARY};
        color: {Colors.TEXT_PRIMARY};
        border: 1px solid {Colors.BORDER};
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: bold;
        min-width: 80px;
    }}
    
    QPushButton:hover {{
        background-color: {Colors.ACCENT};
        border-color: {Colors.BORDER_FOCUS};
    }}
    
    QPushButton:pressed {{
        background-color: {Colors.BG_PRIMARY};
    }}
    
    QPushButton#runButton {{
        background-color: {Colors.HIGHLIGHT};
        border: none;
        color: white;
    }}
    
    QPushButton#runButton:hover {{
        background-color: #ff5a78;
    }}
    
    /* Dock Widgets */
    QDockWidget {{
        color: {Colors.TEXT_PRIMARY};
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
    }}
    
    QDockWidget::title {{
        background-color: {Colors.BG_SECONDARY};
        padding: 10px;
        font-weight: bold;
    }}
    
    /* Scroll Areas */
    QScrollArea {{
        background-color: {Colors.BG_PRIMARY};
        border: none;
    }}
    
    QScrollBar:vertical {{
        background-color: {Colors.BG_SECONDARY};
        width: 10px;
        border-radius: 5px;
    }}
    
    QScrollBar::handle:vertical {{
        background-color: {Colors.BORDER};
        border-radius: 5px;
        min-height: 30px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background-color: {Colors.BORDER_FOCUS};
    }}
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    
    /* Form Widgets */
    QLabel {{
        color: {Colors.TEXT_PRIMARY};
        font-size: 10px;
    }}
    
    QLineEdit {{
        background-color: {Colors.BG_TERTIARY};
        color: {Colors.TEXT_PRIMARY};
        border: 1px solid {Colors.BORDER};
        border-radius: 4px;
        padding: 6px 10px;
    }}
    
    QLineEdit:focus {{
        border-color: {Colors.BORDER_FOCUS};
    }}
    
    QDoubleSpinBox, QSpinBox {{
        background-color: {Colors.BG_TERTIARY};
        color: {Colors.TEXT_PRIMARY};
        border: 1px solid {Colors.BORDER};
        border-radius: 4px;
        padding: 4px 8px;
    }}
    
    QDoubleSpinBox:focus, QSpinBox:focus {{
        border-color: {Colors.BORDER_FOCUS};
    }}
    
    QComboBox {{
        background-color: {Colors.BG_TERTIARY};
        color: {Colors.TEXT_PRIMARY};
        border: 1px solid {Colors.BORDER};
        border-radius: 4px;
        padding: 6px 10px;
        min-width: 100px;
    }}
    
    QComboBox:hover {{
        border-color: {Colors.BORDER_FOCUS};
    }}
    
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    
    QComboBox QAbstractItemView {{
        background-color: {Colors.BG_TERTIARY};
        color: {Colors.TEXT_PRIMARY};
        selection-background-color: {Colors.ACCENT};
        border: 1px solid {Colors.BORDER};
    }}
    
    QCheckBox {{
        color: {Colors.TEXT_PRIMARY};
        spacing: 8px;
    }}
    
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 1px solid {Colors.BORDER};
        background-color: {Colors.BG_TERTIARY};
    }}
    
    QCheckBox::indicator:checked {{
        background-color: {Colors.BORDER_FOCUS};
        border-color: {Colors.BORDER_FOCUS};
    }}
    
    /* Message Box */
    QMessageBox {{
        background-color: {Colors.BG_PRIMARY};
    }}
    
    QMessageBox QLabel {{
        color: {Colors.TEXT_PRIMARY};
    }}
    
    /* Status Bar */
    QStatusBar {{
        background-color: {Colors.BG_SECONDARY};
        color: {Colors.TEXT_SECONDARY};
        border-top: 1px solid {Colors.BORDER};
    }}
    
    /* Graphics View */
    QGraphicsView {{
        border: none;
        background-color: {Colors.BG_PRIMARY};
    }}
    
    /* Tooltips */
    QToolTip {{
        background-color: {Colors.BG_CARD};
        color: {Colors.TEXT_PRIMARY};
        border: 1px solid {Colors.BORDER};
        border-radius: 4px;
        padding: 6px;
    }}
    
    /* Menu */
    QMenu {{
        background-color: {Colors.BG_SECONDARY};
        color: {Colors.TEXT_PRIMARY};
        border: 1px solid {Colors.BORDER};
        border-radius: 6px;
        padding: 4px;
    }}
    
    QMenu::item {{
        padding: 8px 24px;
        border-radius: 4px;
    }}
    
    QMenu::item:selected {{
        background-color: {Colors.ACCENT};
    }}
    
    QMenuBar {{
        background-color: {Colors.BG_SECONDARY};
        color: {Colors.TEXT_PRIMARY};
    }}
    
    QMenuBar::item:selected {{
        background-color: {Colors.ACCENT};
    }}
"""


def get_property_panel_stylesheet() -> str:
    """Return stylesheet for the property editor panel."""
    return f"""
    QWidget#propertyPanel {{
        background-color: {Colors.BG_PRIMARY};
    }}
    
    QFrame#propertyCard {{
        background-color: {Colors.BG_CARD};
        border: 1px solid {Colors.BORDER};
        border-radius: 8px;
        padding: 12px;
        margin: 4px 0;
    }}
    
    QLabel#sectionHeader {{
        color: {Colors.TEXT_SECONDARY};
        font-size: 9px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 8px 0 4px 0;
    }}
    
    QLabel#nodeTitle {{
        color: {Colors.TEXT_PRIMARY};
        font-size: 14px;
        font-weight: bold;
        padding-bottom: 8px;
    }}
    
    QLabel#nodeType {{
        color: {Colors.HIGHLIGHT};
        font-size: 10px;
        padding-bottom: 12px;
    }}
"""


# =============================================================================
# GRADIENT HELPERS
# =============================================================================

def create_node_gradient(node_type: str, width: float, height: float) -> QLinearGradient:
    """Create a gradient brush for a node based on its type."""
    colors = Colors.get_node_colors(node_type)
    gradient = QLinearGradient(0, 0, width, height)
    gradient.setColorAt(0, QColor(colors[0]))
    gradient.setColorAt(1, QColor(colors[1]))
    return gradient


def create_header_gradient(node_type: str, width: float) -> QLinearGradient:
    """Create a gradient for the node header."""
    colors = Colors.get_node_colors(node_type)
    gradient = QLinearGradient(0, 0, width, 0)
    gradient.setColorAt(0, QColor(colors[0]))
    gradient.setColorAt(1, QColor(colors[1]).darker(110))
    return gradient


# =============================================================================
# ICON HELPERS (Simple Unicode-based icons)
# =============================================================================

class Icons:
    """Unicode icons for toolbar and UI elements."""
    
    HOUSEHOLD = "👥"
    POLICY = "🏛️"
    MARKET = "📊"
    FISCAL = "💰"
    RUN = "▶"
    ADD = "+"
    DELETE = "✕"
    SETTINGS = "⚙"
    SAVE = "💾"
    LOAD = "📂"
    ZOOM_IN = "🔍+"
    ZOOM_OUT = "🔍-"
    FIT = "⊡"
