import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QAction, QLabel, QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QThread, pyqtSignal


from PyQt5.QtWidgets import QMainWindow,QVBoxLayout,QHBoxLayout, QWidget

from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from ultralytics import YOLO

from main_ui import Ui_MainWindow

# Problems when changing camera while the camera is on

class InferenceProcessor(QThread):
    inference_done = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.frame = None
        
        self.YOLO_model = None
        self.ResNet_model = None
        
        self.YOLO_model_name = "None"
        self.ResNet_model_name = "None"
        
        self.class_Name = ['20','30','40','50','60','70','80','90','100','120',
            'Crosswalk','No Overtakes','Stop','Traffic Light - Green'
            ,'Traffic Light - Red','Yield']

    def run(self):
        while self.running:
            if self.frame is not None:
                processed_frame = self.process_frame(self.frame)
                self.inference_done.emit(processed_frame)
    # Loading Models
    def load_ResNet(self, model_path):
        self.ResNet_model = load_model(model_path)
    
    def load_YOLO(self, model_path):
        self.YOLO_model = YOLO(model_path)
        
    # YOLO Detection Phase
    def process_frame(self, frame):
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.YOLO_model.predict(source=frame, save=False, conf=0.25, show=False)

        # Cropping for ResNet Identification Phase
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cropped = frame[y1:y2, x1:x2]
                
                # Process the cropped ROI with ResNet50V2
                if cropped.size > 0: 
                    resized = cv2.resize(cropped, (224, 224))
                    # array = img_to_array(resized)  # Convert to array
                    # array = np.expand_dims(array, axis=0)  # Add batch dimension
                    array = np.expand_dims(resized, axis=0)
                    array = preprocess_input(array)  # Preprocess for ResNet50V2

                    # ResNet50V2 Prediction
                    # predictions = self.ResNet_model.predict(array)
                    # print(f'Predicted: {self.class_Name[predictions.argmax()]} - {(predictions.max()*100):.3f}%')# Output predictions to the terminal
                    predictions = self.ResNet_model(array, training = False).numpy()
                    print(f'Predicted: {self.class_Name[predictions.argmax()]} - {(predictions.max()*100):.3f}%') # Output predictions to the terminal
        
        return frame

class mainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mainWindow,self).__init__()
        self.setupUi(self)
        
        self.Inference = InferenceProcessor()
        self.Inference.inference_done.connect(self.Display_output)
        
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
        self.CamSel = False
        self.menuCamera.clear()
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            action = QAction(f"Camera {index}",self)
            
            action.triggered.connect(lambda checked, no= index : self.Set_Cam(no))
            
            self.menuCamera.addAction(action)
            
            cap.release()
            index += 1
        
        self.menuCamera.addSeparator()
        self.menuCamera.addAction(self.actionRefresh)

    # Functions for camera action buttons / Sets camera
    def Set_Cam(self, no):
        self.CamSel = True
        print(f'cam set = {no}')
        self.Capture = cv2.VideoCapture(no)
        if self.Capture.isOpened():
            self.timer.start(60) # Refresh rate in ms

    # This always runs
    def Camera_Output(self):
            ret, frame = self.Capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if ret:
                if self.IsInfeOn and self.Inference:
                    # frame = self.Inference.process_frame(frame)
                    self.Inference.frame = frame
                else:
                    self.Display_output(frame)
            else:
                print("Retrival of image problem with camera!")
                
    def Display_output(self, frame):
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
        if model_path and self.Inference:
            self.Inference.load_ResNet(model_path)
            self.Inference.ResNet_model_name = f"{model_path.split('/')[-1]}"
            self.ResNet_Label.setText("ResNet50V2 Model Used: " + self.Inference.ResNet_model_name)
            self.statusbar.showMessage("Successfully loaded " +  self.Inference.ResNet_model_name + " as the ResNet50V2 model. ", 7000)
            if self.IsInfeOn:
                self.status_Label.setText('Inference Mode: Off')
            else:
                self.status_Label.setText('Inference Mode: On')

    def check_models(self):
        if not self.Inference.YOLO_model or not self.Inference.ResNet_model:
            if self.Inference.YOLO_model:
                self.show_warning("Please load a YOLO Model before proceeding! (*.pt)")
            else:
                self.show_warning("Please load a ResNEt50v2 Model before proceeding! (*.keras)")
            return False
        else:
            return True
        
    def enable_Inference(self):
        if self.CamSel:
            if self.check_models():
                self.IsInfeOn = not self.IsInfeOn
                self.Inference_Button.setText("Disable YOLO_ResNet50V2 Inference" if self.IsInfeOn else "Enable YOLO_ResNet50V2 Inference")
                self.status_Label.setText("Inference Mode: On" if self.IsInfeOn else "Inference Mode: Off")
                
                # run the seperate thread
                if self.IsInfeOn:
                    # self.timer.stop()
                    self.Inference.running = True
                    self.Inference.start()
                else:
                    self.Inference.running = False
                    self.Inference.quit()
                    # self.timer.start(60)
                
            else:
                self.show_warning("Please load both YOLO and ResNet models before enabling inference.")
        else:
            self.show_warning("Please select a camera before enabling inference.")
    
    def closeEvent(self, event):
        print("Resources released.")
        self.timer.stop()
        if self.Capture:
            self.Capture.release()
        super().closeEvent(event)
    
    def show_warning(self, message):
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
