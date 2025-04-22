import logging
import cv2
from camera_manager import CameraManager
from object_detector import ObjectDetector
from movement_tracker import MovementTracker
from security_server import SecurityServer

def main():
    # הגדרת לוגר
    logging.basicConfig(
        filename="server.log",
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # אתחול המרכיבים
        camera = CameraManager()
        detector = ObjectDetector()
        tracker = MovementTracker()
        server = SecurityServer()
        
        # אתחול המצלמה
        if not camera.initialize_camera():
            logger.error("Failed to initialize camera")
            return
            
        # הפעלת השרת
        server.start()
        
        # לולאה ראשית
        prev_results = None
        while camera.is_running:
            # קריאת פריים מהמצלמה
            success, frame = camera.read_frame()
            if not success:
                continue
                
            # זיהוי אובייקטים
            results = detector.detect_objects(frame)
            if results:
                # חילוץ תיבות של אנשים
                person_boxes = detector.get_person_boxes(results)
                
                # מעקב אחר תנועה
                tracker.track_movement(frame, person_boxes, prev_results)
                
                # שמירת התוצאות לפריים הבא
                prev_results = results
                
            # שליחת הפריים לכל הלקוחות
            server.broadcast_frame(frame)
            
            # בדיקה להפסקת התוכנית
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
    finally:
        # ניקוי משאבים
        camera.release()
        server.stop()
        cv2.destroyAllWindows()
        logger.info("Server stopped")

if __name__ == '__main__':
    main() 