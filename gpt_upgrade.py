# Smart Human Tracker with YOLO, Encryption, Socket Communication, Threading, and OOP

import cv2
import logging
import socket
import json
import threading
from cryptography.fernet import Fernet
from ultralytics import YOLO

# ======================= Global Settings =======================
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
MODEL_PATH = "yolov8n.pt"
SERVER_ADDRESS = ("localhost", 12345)
FERNET_KEY = Fernet.generate_key()


# ======================= Camera Handler =======================
class CameraHandler:
    def __init__(self, width, height):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_frame(self):
        success, frame = self.cam.read()
        return frame if success else None

    def release(self):
        self.cam.release()
        cv2.destroyAllWindows()


# ======================= Person Tracker =======================
class PersonTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.names = self.model.names
        self.prev_results = None

    def track(self, frame):
        return self.model.track(frame, persist=True)

    def get_direction(self, current_box, prev_box):
        x = (current_box.xyxy[0][0] + current_box.xyxy[0][2]) / 2
        prev_x = (prev_box.xyxy[0][0] + prev_box.xyxy[0][2]) / 2
        return ">" if x > prev_x else "<"

    def get_prev_box(self, prev_results, box_id):
        for box in prev_results[0].boxes:
            if box.cls.tolist()[0] == 0 and box.id == box_id:
                return box
        return None


# ======================= Direction Manager =======================
class DirectionManager:
    def __init__(self):
        self.going_right_ids = set()
        self.going_left_ids = set()

    def assign(self, box):
        x = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
        if x < FRAME_WIDTH - x:
            self.going_left_ids.add(int(box.id))
            return "<"
        else:
            self.going_right_ids.add(int(box.id))
            return ">"

    def expected_direction(self, box_id):
        if box_id in self.going_right_ids:
            return ">"
        elif box_id in self.going_left_ids:
            return "<"
        return None


# ======================= Secure Sender =======================
class SecureSender:
    def __init__(self, key):
        self.cipher = Fernet(key)

    def send(self, data: dict):
        thread = threading.Thread(target=self._send_thread, args=(data,))
        thread.start()

    def _send_thread(self, data):
        try:
            with socket.socket() as s:
                s.connect(SERVER_ADDRESS)
                encrypted = self.cipher.encrypt(json.dumps(data).encode())
                s.send(encrypted)
        except Exception as e:
            print("[ERROR] Failed to send data:", e)


# ======================= Drawing Function =======================
def draw_box(frame, box, color, label):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# ======================= Direction Processing =======================
def process_direction(tracker, manager, sender, frame, box):
    box_id = int(box.id)
    expected = manager.expected_direction(box_id)
    prev_box = tracker.get_prev_box(tracker.prev_results, box_id)

    if expected and prev_box:
        actual = tracker.get_direction(box, prev_box)
        valid = (actual == expected)
        color = (0, 255, 0) if valid else (0, 0, 255)
        label = f"ID: {box_id} Dir: {actual} {'✓' if valid else '✗'}"
        draw_box(frame, box, color, label)
        sender.send({"id": box_id, "dir": actual, "valid": valid})

    elif expected is None:
        dir_assigned = manager.assign(box)
        draw_box(frame, box, (255, 0, 0), f"ID: {box_id} Dir: {dir_assigned} ?")


# ======================= Main Application =======================
def main():
    logging.basicConfig(filename="tracker.log", filemode='w', level=logging.DEBUG)

    camera = CameraHandler(FRAME_WIDTH, FRAME_HEIGHT)
    tracker = PersonTracker(MODEL_PATH)
    manager = DirectionManager()
    sender = SecureSender(FERNET_KEY)

    tracker.prev_results = tracker.track(camera.get_frame())

    while True:
        frame = camera.get_frame()
        if frame is None:
            break

        results = tracker.track(frame)

        for box in results[0].boxes:
            if box.id and box.cls.tolist()[0] == 0:
                process_direction(tracker, manager, sender, frame, box)

        cv2.imshow("Smart Tracker", frame)
        tracker.prev_results = results
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()


if __name__ == '__main__':
    main()
