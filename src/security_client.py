import socket
import threading
import logging
import json
import base64
import cv2
import numpy as np
from cryptography.fernet import Fernet

class SecurityClient:
    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        self.client_socket = None
        self.cipher_suite = None
        self.is_running = False
        self.logger = logging.getLogger(__name__)
        
    def connect(self):
        """Connect to the security server"""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.host, self.port))
            self.is_running = True
            
            # Receive encryption key from server
            key = self.client_socket.recv(4096)
            self.cipher_suite = Fernet(key)
            
            # Start receiving data in a separate thread
            receive_thread = threading.Thread(target=self._receive_data)
            receive_thread.daemon = True
            receive_thread.start()
            
            self.logger.info(f"Connected to server at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to server: {str(e)}")
            return False
            
    def _receive_data(self):
        """Receive and process data from server"""
        while self.is_running:
            try:
                data = self.client_socket.recv(4096)
                if not data:
                    break
                    
                # Decrypt the data
                decrypted_data = self.cipher_suite.decrypt(data)
                message = json.loads(decrypted_data.decode('utf-8'))
                
                if message['type'] == 'frame':
                    # Convert base64 to image
                    frame_bytes = base64.b64decode(message['data'])
                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Display the frame
                    cv2.imshow("Security Camera", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop()
                        
            except Exception as e:
                self.logger.error(f"Error receiving data: {str(e)}")
                break
                
    def send_command(self, command):
        """Send a command to the server"""
        try:
            message = {
                'type': 'command',
                'data': command
            }
            
            encrypted_message = self.cipher_suite.encrypt(
                json.dumps(message).encode('utf-8')
            )
            
            self.client_socket.send(encrypted_message)
            
        except Exception as e:
            self.logger.error(f"Error sending command: {str(e)}")
            
    def stop(self):
        """Stop the security client"""
        self.is_running = False
        if self.client_socket:
            self.client_socket.close()
        cv2.destroyAllWindows()
        self.logger.info("Client stopped") 