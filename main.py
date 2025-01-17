import sys
import cv2
import time
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QAction, QLabel, QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from ultralytics import YOLO

from main_ui import Ui_MainWindow

class InferenceProcessor:
    def __init__(self):
        self.YOLO_model = None
        self.ResNet_model = None
        
        self.YOLO_model_name = "None"
        self.ResNet_model_name = "None"
        
    # Loading Models
    def load_ResNet(self, model_path):
        self.ResNet_model = load_model(model_path)
    
    def load_YOLO(self, model_path):
        self.YOLO_model = YOLO(model_path)
        
    # YOLO Detection Phase
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.YOLO_model.predict(source=rgb_frame, save=False, conf=0.25, show=False)

        # Cropping for ResNet Identification Phase
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cropped = frame[y1:y2, x1:x2]

                # Process the cropped ROI with ResNet50V2
                if cropped.size > 0: 
                    resized = cv2.resize(cropped, (224, 224))  # Resize to ResNet50V2 input size
                    array = img_to_array(resized)  # Convert to array
                    array = np.expand_dims(array, axis=0)  # Add batch dimension
                    array = preprocess_input(array)  # Preprocess for ResNet50V2

                    # ResNet50V2 Prediction
                    predictions = self.resnet.predict(array)
                    print(predictions)  # Output predictions to the terminal

        return rgb_frame

class mainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mainWindow,self).__init__()
        self.setupUi(self)
        
        self.Inference = InferenceProcessor()
        self.Capture = None
        self.IsInfeOn = False
        self.CamSel = False
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.Camera_Output)
        
        self.YOLO_Label = QLabel("YOLO Model Used: " + self.Inference.YOLO_model_name)
        self.ResNet_Label = QLabel("ResNet50V2 Model Used: " + self.Inference.ResNet_model_name)
        self.status_Label = QLabel("Inference Mode: Off")
        
        # Status Bar Shenanigans
        
        self.statusbar.addPermanentWidget(self.YOLO_Label)
        self.statusbar.addPermanentWidget(self.ResNet_Label)
        self.statusbar.addWidget(self.status_Label)
        
        # Action Bar
        self.actionLoad_ResNet.triggered.connect(self.load_ResNet_model)
        self.actionLoad_YOLO.triggered.connect(self.load_YOLO_model)
        self.actionRefresh.triggered.connect(self.Refresh_Cams)
        
        # Buttons
        self.Inference_Button.clicked.connect(self.enable_Inference)
        
        self.Refresh_Cams()
        self.show()
        
    # Placing action buttons for each camera
    def Refresh_Cams(self):
        self.menuCamera.clear()
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            action = QAction(f"Camera {index}",self)
            
            action.triggered.connect(lambda checked, no= index: self.Set_Cam(no))
            
            self.menuCamera.addAction(action)
            
            cap.release()
            index += 1
        
        self.menuCamera.addSeparator()
        self.menuCamera.addAction(self.actionRefresh)

    # Functions for camera action buttons / Sets camera
    def Set_Cam(self, no):
        self.Capture = cv2.VideoCapture(no)
        if self.Capture.isOpened():
            self.timer.start(60) # Refresh rate in ms

    def Camera_Output(self):
            ret, frame = self.Capture.read()
            if ret:
                if self.yolo_enabled and self.inference_processor:
                    frame = self.inference_processor.process_frame(frame)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image = QImage(
                    frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888
                )
                pixmap = QPixmap.fromImage(image)
                self.Vid_label.setPixmap(pixmap)

    # Load YOLO Model
    # WIP setting inference back on and off when changing models
    def load_YOLO_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Select a YOLO Model", "", "YOLO Model (*.pt)")
        if model_path:
            self.Inference.load_YOLO(model_path)
            self.Inference.YOLO_model_name = f"{model_path.split('/')[-1]}"
            self.YOLO_Label.setText("YOLO Model Used: " + self.Inference.YOLO_model_name)
            self.statusbar.showMessage("Successfully loaded " +  self.Inference.YOLO_model_name + " as the YOLOv8 model. ")
            if self.IsInfeOn:
                self.status_Label.setText('Inference Mode: Off (hi)')
            else:
                self.status_Label.setText('Inference Mode: On')

    # Load ResNet50v2 Model
    # WIP setting inference back on and off when changing models
    def load_ResNet_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Select a ResNet50V2 Model", "", "Keras Model (*.keras)")
        if model_path and self.inference_processor:
            self.Inference.load_ResNet(model_path)
            self.Inference.ResNet_model_name = f"{model_path.split('/')[-1]}"
            self.ResNet_Label.setText("ResNet50V2 Model Used: " + self.Inference.ResNet_model_name)
            self.statusbar.showMessage("Successfully loaded " +  self.Inference.ResNet_model_name + " as the ResNet50V2 model. ", 7000)
            if self.IsInfeOn:
                self.status_Label.setText('Inference Mode: Off (hi)')
            else:
                self.status_Label.setText('Inference Mode: On')

    def enable_Inference(self):
        if self.CamSel:
            if self.Inference.YOLO_model and self.Inference.ResNet_model:
                self.IsInfeOn = not self.IsInfeOn
                self.Inference_Button.setText("Disable YOLO_ResNet50V2 Inference" if self.IsInfeOn else "Enable YOLO_ResNet50V2 Inference")
                self.status_Label.setText("Inference Mode: On" if self.IsInfeOn else "Inference Mode: Off")
            else:
                self.show_warning("Please load both YOLO and ResNet models before enabling inference.")
        else:
            self.show_warning("Please select a camera before enabling inference.")
    
    def show_warning(self, message):
        """Show a warning message to the user."""
        warning = QMessageBox(self)
        warning.setIcon(QMessageBox.Warning)
        warning.setWindowTitle("Warning")
        warning.setText(message)
        warning.setStandardButtons(QMessageBox.Ok)
        warning.exec_()
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    Win = mainWindow()

    sys.exit(app.exec())
