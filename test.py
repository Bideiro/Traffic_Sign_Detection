from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QMessageBox
import random

class DynamicMenuBar(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic Menu Bar with Refresh")
        self.setGeometry(100, 100, 600, 400)

        # Create the menu bar
        self.menu_bar = self.menuBar()

        # Create the dynamic menu
        self.dynamic_menu = self.menu_bar.addMenu("Dynamic Menu")

        # Add the Refresh button
        self.refresh_action = QAction("Refresh", self)
        self.refresh_action.triggered.connect(self.refresh_menu)
        self.dynamic_menu.addAction(self.refresh_action)

        # Add initial actions
        self.refresh_menu()

    def refresh_menu(self):
        """Clears the menu and adds a random number of new actions."""
        # Clear all actions except the Refresh button
        self.dynamic_menu.clear()
        self.dynamic_menu.addAction(self.refresh_action)

        # Generate a random number of actions
        num_actions = random.randint(3, 10)
        for i in range(1, num_actions + 1):
            action = QAction(f"Item {i}", self)
            action.triggered.connect(lambda _, name=f"Item {i}": self.show_message(name))
            self.dynamic_menu.addAction(action)

        # Show a status update
        self.statusBar().showMessage(f"Menu refreshed with {num_actions} items.")

    def show_message(self, name):
        """Displays a message when an action is clicked."""
        QMessageBox.information(self, "Action Triggered", f"You clicked: {name}")

if __name__ == "__main__":
    app = QApplication([])

    window = DynamicMenuBar()
    window.show()

    app.exec_()
