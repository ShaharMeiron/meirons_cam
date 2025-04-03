import cv2
from ultralytics import YOLO


model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Run YOLO tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, )
        print("**************************************************************************************************")
        print(results)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        for box in results[0].boxes:
            print(box.xyxy.tolist(), box.conf.item(), int(box.cls))  # Coordinates, confidence, class ID

        quit()
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

