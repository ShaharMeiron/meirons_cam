import cv2
from ultralytics import YOLO
import logging
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# if frames are bigger my home camera might not work well
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

model = YOLO("yolov8n.pt")
# model.to("cuda")
names = model.names

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)  # Lower resolution for better performance
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cam.set(cv2.CAP_PROP_FPS, 30)  # Adjust FPS

# colors are in bgr format
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)


def get_first_results():
    if cam.isOpened():
        success, frame = cam.read()

        if success:
            results = model.track(frame, persist=True, )

            for box in results[0].boxes:
                print(f"object: {names[int(box.cls)]}, Confidence: {box.conf.item()}, Coordinates: {box.xyxy.tolist()}")

            annotated_frame = results[0].plot()

            cv2.imshow("YOLO8 Tracking", annotated_frame)
            return results


def get_prev_box(prev_results, box_id):
    for box in prev_results[0].boxes:
        if box.cls.tolist()[0] == 0 and box.id == box_id:
            return box
    return None


def get_direction(box, prev_box):
    x1, y1, x2, y2 = box.xyxy[0]
    x = (x1 + x2) / 2
    x1, y1, x2, y2 = prev_box.xyxy[0]
    prev_x = (x1 + x2) / 2
    if x > prev_x:
        return ">"
    else:
        return "<"


def draw_data(frame, box, color, direction=""):
    logging.debug(f"object: {names[int(box.cls)]}, Confidence: {box.conf.item()}, Coordinates: {box.xyxy[0]}, direction: {direction}")
    header = f"conf: {round(int(100 * box.conf.item()))}, direction: {direction}, id: {int(box.id)}"
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = round(x1.item()), round(y1.item()), round(x2.item()), round(y2.item())
    cv2.rectangle(frame, (x1, y1), (x2, y2), color)
    cv2.putText(frame, header, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color)


def choose_box_side(box, left, right):
    x1, y1, x2, y2 = box.xyxy[0]
    x = (x1 + x2) / 2
    if x < FRAME_WIDTH - x:
        left.add(int(box.id))
        logging.debug("adding person to left")
    else:
        right.add(int(box.id))
        logging.debug("adding person to right")


def check_direction(box, right_direction, prev_results, frame):

    prev_box = get_prev_box(prev_results, box.id)
    if prev_box:
        direction = get_direction(box, prev_box)
        if right_direction == direction:
            draw_data(frame, box, GREEN, direction)
        else:
            draw_data(frame, box, RED, direction)


def main():
    going_right_ids = set()
    going_left_ids = set()
    prev_results = get_first_results()

    while cam.isOpened():
        success, frame = cam.read()

        if success:
            results = model.track(frame, persist=True, )

            for box in results[0].boxes:
                logging.debug("current box:")
                logging.debug(box)
                if box.id:
                    if box.cls.tolist()[0] == 0:  # if person
                        if int(box.id) in going_right_ids:
                            check_direction(box, ">", prev_results, frame)
                        elif int(box.id) in going_left_ids:
                            check_direction(box, "<", prev_results, frame)
                        else:
                            choose_box_side(box, going_right_ids, going_left_ids)
                            draw_data(frame, box, BLUE)

            cv2.imshow("YOLO8 Tracking", frame)
            prev_results = results
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(filename="cam.log", filemode='w', level=logging.DEBUG)
    main()
