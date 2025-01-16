import sys
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt
import cv2
from ultralytics import YOLO
import numpy as np

class YOLOApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = YOLO("Completed Models/YOLOv8s(lightweight)_e10_detect_11-25-24/weights/best.pt")  # Load YOLO model
        self.capture = cv2.VideoCapture(0)  # Initialize webcam capture
        self.timer = QTimer(self)  # Timer for updating frames
        self.timer.timeout.connect(self.update_frame)  # Call update_frame on timeout
        self.timer.start(30)  # Update every 30ms (~33 FPS)

    def initUI(self):
        self.setWindowTitle("YOLO Live Webcam")
        self.setGeometry(200, 200, 800, 600)

        # Layout
        layout = QVBoxLayout()

        # QLabel for displaying video feed
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # Stop/Start Button
        self.toggle_button = QPushButton("Stop Webcam", self)
        self.toggle_button.clicked.connect(self.toggle_webcam)
        layout.addWidget(self.toggle_button)

        # Set layout
        self.setLayout(layout)

        self.is_running = True  # Webcam running flag

    def toggle_webcam(self):
        if self.is_running:
            self.timer.stop()
            self.capture.release()
            self.toggle_button.setText("Start Webcam")
        else:
            self.capture = cv2.VideoCapture(0)  # Reinitialize capture
            self.timer.start(30)
            self.toggle_button.setText("Stop Webcam")
        self.is_running = not self.is_running

    def update_frame(self):
        if not self.capture.isOpened():
            return

        # Capture frame from webcam
        ret, frame = self.capture.read()
        if not ret:
            return

        # YOLO Inference
        results = self.model(frame)  # Perform YOLO detection

        # Annotate frame with detections
        annotated_frame = results[0].plot()  # YOLO annotated frame

        # Convert image to QPixmap for display
        h, w, ch = annotated_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(annotated_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Display in QLabel
        self.video_label.setPixmap(pixmap)
        self.video_label.setScaledContents(True)

    def closeEvent(self, event):
        # Release resources on close
        if self.capture.isOpened():
            self.capture.release()
        self.timer.stop()
        event.accept()
            
    # def update_frame(self):
    #     if not self.capture.isOpened():
    #         return

    #     # Capture frame from webcam
    #     ret, frame = self.capture.read()
    #     if not ret:
    #         return

    #     # YOLO Inference
    #     results = self.model(frame)  # Perform YOLO detection

    #     # Annotate frame manually (without class name or confidence)
    #     annotated_frame = frame.copy()
    #     for box in results[0].boxes:  # Access detected boxes
    #         x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
    #         color = (0, 255, 0)  # Green bounding box
    #         thickness = 2
    #         cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

    #     # Convert from BGR to RGB
    #     annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    #     # Convert image to QPixmap for display
    #     h, w, ch = annotated_frame.shape
    #     bytes_per_line = ch * w
    #     qt_image = QImage(annotated_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
    #     pixmap = QPixmap.fromImage(qt_image)

    #     # Display in QLabel
    #     self.video_label.setPixmap(pixmap)
    #     self.video_label.setScaledContents(True)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec_())
