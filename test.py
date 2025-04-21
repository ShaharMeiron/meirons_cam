import cv2

# Initialize the camera
cam = cv2.VideoCapture(0)

# Get the current resolution
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Current camera resolution: {frame_width}x{frame_height}")

# Don't forget to release the camera when done
cam.release()
