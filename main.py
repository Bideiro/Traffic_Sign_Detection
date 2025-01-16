import sys
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QAction, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer


from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

from ultralytics import YOLO

from main_ui import Ui_MainWindow

# class InferenceProcessor:
#     """Class to handle YOLO inference, crop bounding boxes, and process them with ResNet50V2."""
#     def __init__(self, model):
#         self.model = model
#         self.resnet = ResNet50V2(weights="imagenet")  # Load pre-trained ResNet50V2

#     def process_frame(self, frame):
#         """Run YOLO inference, crop bounding boxes, and process them with ResNet50V2."""
#         # Convert BGR to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Run YOLO model on the frame
#         results = self.model.predict(source=rgb_frame, save=False, conf=0.25, show=False)

#         # Draw bounding boxes and process cropped regions
#         for result in results:
#             for box in result.boxes.xyxy:
#                 x1, y1, x2, y2 = map(int, box[:4])

#                 # Draw bounding box
#                 cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

#                 # Crop the region of interest (ROI)
#                 cropped = frame[y1:y2, x1:x2]

#                 # Process the cropped ROI with ResNet50V2
#                 if cropped.size > 0:  # Ensure the crop is valid
#                     resized = cv2.resize(cropped, (224, 224))  # Resize to ResNet50V2 input size
#                     array = img_to_array(resized)  # Convert to array
#                     array = np.expand_dims(array, axis=0)  # Add batch dimension
#                     array = preprocess_input(array)  # Preprocess for ResNet50V2

#                     # Predict using ResNet50V2
#                     predictions = self.resnet.predict(array)
#                     print(predictions)  # Output predictions to the terminal

#         return rgb_frame

class mainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mainWindow,self).__init__()
        self.setupUi(self)
        
        self.capture = None
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.Camera_Output)
        
        self.statusbar.showMessage('hi', 2000)
        self.statusbar.showMessage('yo')
        # Action Bar
        self.On_cam.clicked.connect(self.Cam_on)
        
        # Buttons
        self.actionRefresh.triggered.connect(self.Refresh_Cams)
        
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
    
    # Functions for camera action buttons / Sets which camera to use
    def Set_Cam(self, no):
        self.capture = cv2.VideoCapture(no)
        if self.capture.isOpened():
            self.timer.start(60) # Refresh rate in ms
            
    def Camera_Output(self):
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
                self.Vid_label.setPixmap(pixmap)
    
    # def process_frame_with_yolo(self, frame):
    #     """Process the frame with YOLOv8 model."""

    #     # Run YOLO model on the frame
    #     results = self.model.predict(source=frame, save=False, conf=0.25, show=False)

    #     # Draw bounding boxes
    #     for result in results:
    #         for box in result.boxes.xyxy:
    #             x1, y1, x2, y2 = map(int, box[:4])
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    #     return frame
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    Win = mainWindow()

    sys.exit(app.exec())
