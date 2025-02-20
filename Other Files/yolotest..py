from ultralytics import YOLO
# from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

# model = YOLO("C:/Users/dei/Documents/Programming/YOLOv8s(TrafficSignNou)_e10_detect_11-30-24.pt")
model = YOLO("yolov8s.pt")

results = model.predict(source="1", show=True)

print(results)