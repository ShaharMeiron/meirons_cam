from ultralytics import YOLO
import logging
import numpy as np

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.names = self.model.names
        self.logger = logging.getLogger(__name__)
        
    def detect_objects(self, frame):
        """Detect objects in the frame using YOLO"""
        try:
            results = self.model.track(frame, persist=True)
            return results
        except Exception as e:
            self.logger.error(f"Error in object detection: {str(e)}")
            return None
            
    def get_person_boxes(self, results):
        """Extract person boxes from detection results"""
        if not results or len(results) == 0:
            return []
            
        person_boxes = []
        for box in results[0].boxes:
            if box.cls.tolist()[0] == 0:  # if person
                person_boxes.append(box)
        return person_boxes
        
    def get_box_center(self, box):
        """Calculate the center point of a box"""
        x1, y1, x2, y2 = box.xyxy[0]
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        return x, y 