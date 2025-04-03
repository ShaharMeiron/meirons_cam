import cv2
from ultralytics import YOLO
import logging


model = YOLO("yolov8n.pt")
names = model.names

cap = cv2.VideoCapture(0)


def get_first_results():
    if cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.track(frame, persist=True, )

            for box in results[0].boxes:
                print(f"object: {names[int(box.cls)]}, Confidence: {box.conf.item()}, Coordinates: {box.xyxy.tolist()}")

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            cv2.imshow("YOLO8 Tracking", annotated_frame)
            return results


def main():
    prev_results = get_first_results()
    directions = {}
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.track(frame, persist=True, )

            # calculate direction - up or down
            for box in results[0].boxes:
                if box.cls == '0':  # if human
                    current_box_id = box.id
                    for prev_box in prev_results[0].boxes:
                        if box.cls == '0' and current_box_id == prev_box.id:
                            if current_box_id not in directions:
                                directions[current_box_id] = box.xyxy.tolist()



                # print(f"object: {names[int(box.cls)]}, Confidence: {box.conf.item()}, Coordinates: {box.xyxy.tolist()}")

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            cv2.imshow("YOLO8 Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
