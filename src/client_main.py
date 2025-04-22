import logging
import cv2
from security_client import SecurityClient

def main():
    # הגדרת לוגר
    logging.basicConfig(
        filename="client.log",
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # אתחול הלקוח
        client = SecurityClient()
        
        # התחברות לשרת
        if not client.connect():
            logger.error("Failed to connect to server")
            return
            
        # לולאה ראשית
        while client.is_connected:
            # קבלת פריים מהשרת
            frame = client.receive_frame()
            if frame is None:
                continue
                
            # הצגת הפריים
            cv2.imshow('Security Camera', frame)
            
            # בדיקה להפסקת התוכנית
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
    finally:
        # ניקוי משאבים
        client.disconnect()
        cv2.destroyAllWindows()
        logger.info("Client stopped")

if __name__ == '__main__':
    main() 