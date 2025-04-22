import cv2
import numpy as np

class ObjectTracker:
    def __init__(self):
        self.next_id = 0
        self.tracked_objects = {}

    def update(self, detections):
        new_tracked = {}
        for cx, cy, x, y, w, h in detections:
            matched = False
            for obj_id, (prev_x, prev_y) in self.tracked_objects.items():
                if abs(cx - prev_x) < 30 and abs(cy - prev_y) < 30:
                    new_tracked[obj_id] = (cx, cy)
                    yield obj_id, x, y, w, h
                    matched = True
                    break

            if not matched:
                new_tracked[self.next_id] = (cx, cy)
                yield self.next_id, x, y, w, h
                self.next_id += 1

        self.tracked_objects = new_tracked

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def detect_motion(frame, background):
    diff = cv2.absdiff(background, frame)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    return thresh

def extract_blobs(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w//2, y + h//2
        detections.append((cx, cy, x, y, w, h))
    return detections

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    _, frame = cap.read()
    background = preprocess_frame(cv2.resize(frame, (320, 240)))
    tracker = ObjectTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (320, 240))
        pre_frame = preprocess_frame(frame)
        thresh = detect_motion(pre_frame, background)
        detections = extract_blobs(thresh)

        for obj_id, x, y, w, h in tracker.update(detections):
            color = (0, 255, 0) if obj_id in tracker.tracked_objects else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"ID {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
