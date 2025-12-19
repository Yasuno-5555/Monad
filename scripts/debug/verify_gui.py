
import sys
from PySide6.QtWidgets import QApplication
from monad.gui.app import NodeEditor

def test_gui_launch():
    app = QApplication(sys.argv)
    window = NodeEditor()
    # If we get here without error, instantiation worked.
    print("[SUCCESS] GUI Instantiated Successfully.")
    # Do not call app.exec() to avoid hanging
    return

if __name__ == "__main__":
    test_gui_launch()
