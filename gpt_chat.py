import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox, QFileDialog
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from ultralytics import YOLO
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

class InferenceProcessor:
    """Class to handle YOLO inference, crop bounding boxes, and process them with ResNet50V2."""
    def __init__(self, model):
        self.model = model
        self.resnet = ResNet50V2(weights="imagenet")  # Load pre-trained ResNet50V2

    def process_frame(self, frame):
        """Run YOLO inference, crop bounding boxes, and process them with ResNet50V2."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO model on the frame
        results = self.model.predict(source=rgb_frame, save=False, conf=0.25, show=False)

        # Draw bounding boxes and process cropped regions
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])

                # Draw bounding box
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Crop the region of interest (ROI)
                cropped = frame[y1:y2, x1:x2]

                # Process the cropped ROI with ResNet50V2
                if cropped.size > 0:  # Ensure the crop is valid
                    resized = cv2.resize(cropped, (224, 224))  # Resize to ResNet50V2 input size
                    array = img_to_array(resized)  # Convert to array
                    array = np.expand_dims(array, axis=0)  # Add batch dimension
                    array = preprocess_input(array)  # Preprocess for ResNet50V2

                    # Predict using ResNet50V2
                    predictions = self.resnet.predict(array)
                    print(predictions)  # Output predictions to the terminal

        return rgb_frame

class WebcamApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam Selector and YOLOv8 with ResNet50V2")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        # Dropdown for available webcams
        self.webcam_selector = QComboBox(self)
        self.layout.addWidget(self.webcam_selector)

        # Button to start the webcam
        self.start_button = QPushButton("Start Webcam", self)
        self.layout.addWidget(self.start_button)

        # Button to enable/disable YOLO inference
        self.yolo_button = QPushButton("Enable YOLO Inference", self)
        self.layout.addWidget(self.yolo_button)

        # Button to load a YOLO model
        self.load_model_button = QPushButton("Load YOLO Model", self)
        self.layout.addWidget(self.load_model_button)

        # Video display label
        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        self.setLayout(self.layout)

        # Initialize variables
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.model = None  # YOLO model will be loaded dynamically
        self.inference_processor = None  # Separate class for inference
        self.yolo_enabled = False

        self.populate_webcams()

        # Connect buttons to their functions
        self.start_button.clicked.connect(self.start_webcam)
        self.yolo_button.clicked.connect(self.toggle_yolo)
        self.load_model_button.clicked.connect(self.load_yolo_model)

    def populate_webcams(self):
        """Search and list available webcams."""
        self.webcam_selector.clear()
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            self.webcam_selector.addItem(f"Webcam {index}")
            cap.release()
            index += 1

    def start_webcam(self):
        """Start the selected webcam and display its feed."""
        selected_index = self.webcam_selector.currentIndex()
        self.capture = cv2.VideoCapture(selected_index)
        if self.capture.isOpened():
            self.timer.start(30)  # Refresh rate in ms

    def toggle_yolo(self):
        """Enable or disable YOLO inference."""
        self.yolo_enabled = not self.yolo_enabled
        self.yolo_button.setText("Disable YOLO Inference" if self.yolo_enabled else "Enable YOLO Inference")

    def load_yolo_model(self):
        """Open a file dialog to load a YOLO model."""
        model_path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "YOLO Model (*.pt)")
        if model_path:
            self.model = YOLO(model_path)
            self.inference_processor = InferenceProcessor(self.model)
            self.load_model_button.setText(f"Model Loaded: {model_path.split('/')[-1]}")

    def update_frame(self):
        """Capture frame from webcam and display it."""
        ret, frame = self.capture.read()
        if ret:
            if self.yolo_enabled and self.inference_processor:
                frame = self.inference_processor.process_frame(frame)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = QImage(
                frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        """Release resources on close."""
        self.timer.stop()
        if self.capture:
            self.capture.release()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WebcamApp()
    window.show() 
    sys.exit(app.exec_())
