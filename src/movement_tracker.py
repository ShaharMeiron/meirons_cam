import logging
import cv2

class MovementTracker:
    def __init__(self, frame_width=320):
        self.frame_width = frame_width
        self.going_right_ids = set()
        self.going_left_ids = set()
        self.prev_boxes = {}
        self.logger = logging.getLogger(__name__)
        
        # Colors in BGR format
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)
        self.BLUE = (255, 0, 0)
        
    def get_direction(self, box, prev_box):
        """Calculate movement direction based on box positions"""
        x1, y1, x2, y2 = box.xyxy[0]
        x = (x1 + x2) / 2
        x1, y1, x2, y2 = prev_box.xyxy[0]
        prev_x = (x1 + x2) / 2
        return ">" if x > prev_x else "<"
        
    def choose_box_side(self, box):
        """Determine which side of the frame the box is on"""
        x1, y1, x2, y2 = box.xyxy[0]
        x = (x1 + x2) / 2
        if x < self.frame_width - x:
            self.going_right_ids.add(int(box.id))
            self.logger.debug("Adding person to left side")
        else:
            self.going_left_ids.add(int(box.id))
            self.logger.debug("Adding person to right side")
            
    def draw_data(self, frame, box, color, direction=""):
        """Draw box and information on the frame"""
        header = f"conf: {round(int(100 * box.conf.item()))}, direction: {direction}, id: {int(box.id)}"
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = round(x1.item()), round(y1.item()), round(x2.item()), round(y2.item())
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color)
        cv2.putText(frame, header, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        
    def track_movement(self, frame, boxes, prev_results):
        """Track movement of detected boxes"""
        for box in boxes:
            if box.id:
                if int(box.id) in self.going_right_ids:
                    self._check_direction(box, ">", prev_results, frame)
                elif int(box.id) in self.going_left_ids:
                    self._check_direction(box, "<", prev_results, frame)
                else:
                    self.choose_box_side(box)
                    self.draw_data(frame, box, self.BLUE)
                    
    def _check_direction(self, box, right_direction, prev_results, frame):
        """Check if box is moving in the correct direction"""
        prev_box = self._get_prev_box(prev_results, box.id)
        if prev_box:
            direction = self.get_direction(box, prev_box)
            color = self.GREEN if right_direction == direction else self.RED
            self.draw_data(frame, box, color, direction)
            
    def _get_prev_box(self, prev_results, box_id):
        """Get previous box position for a given ID"""
        for box in prev_results[0].boxes:
            if box.cls.tolist()[0] == 0 and box.id == box_id:
                return box
        return None 