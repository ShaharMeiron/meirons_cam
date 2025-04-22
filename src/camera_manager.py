import cv2
import logging

class CameraManager:
    def __init__(self, width=320, height=240, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = None
        self.is_running = False
        self.logger = logging.getLogger(__name__)
        
    def initialize_camera(self):
        """Initialize the camera with the specified settings"""
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            self.is_running = True
            self.logger.info("Camera initialized successfully")
            return self.camera.isOpened()
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {str(e)}")
            return False
        
    def read_frame(self):
        """Read a frame from the camera"""
        if not self.camera or not self.camera.isOpened():
            self.logger.warning("Camera is not initialized or not opened")
            return False, None
        try:
            success, frame = self.camera.read()
            if not success:
                self.logger.warning("Failed to read frame from camera")
            return success, frame
        except Exception as e:
            self.logger.error(f"Error reading frame: {str(e)}")
            return False, None
        
    def release(self):
        """Release the camera resources"""
        if self.camera:
            self.camera.release()
            self.camera = None
            self.is_running = False
            self.logger.info("Camera resources released") 