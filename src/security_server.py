import socket
import threading
import logging
import json
import base64
import cv2
import numpy as np
from cryptography.fernet import Fernet

class SecurityServer:
    def __init__(self, host='0.0.0.0', port=5000):
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = []
        self.is_running = False
        self.logger = logging.getLogger(__name__)
        
        # Generate encryption key
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
        
    def start(self):
        """Start the security server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.is_running = True
            self.logger.info(f"Server started on {self.host}:{self.port}")
            
            # Start accepting clients in a separate thread
            accept_thread = threading.Thread(target=self._accept_clients)
            accept_thread.daemon = True
            accept_thread.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {str(e)}")
            
    def _accept_clients(self):
        """Accept new client connections"""
        while self.is_running:
            try:
                client_socket, address = self.server_socket.accept()
                self.logger.info(f"New connection from {address}")
                self.clients.append(client_socket)
                
                # Start handling client in a separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                self.logger.error(f"Error accepting client: {str(e)}")
                
    def _handle_client(self, client_socket, address):
        """Handle communication with a client"""
        try:
            # Send encryption key to client
            client_socket.send(self.key)
            
            while self.is_running:
                # Receive data from client
                data = client_socket.recv(4096)
                if not data:
                    break
                    
                # Decrypt and process the data
                decrypted_data = self.cipher_suite.decrypt(data)
                # Process the data as needed
                
        except Exception as e:
            self.logger.error(f"Error handling client {address}: {str(e)}")
        finally:
            self.clients.remove(client_socket)
            client_socket.close()
            
    def broadcast_frame(self, frame):
        """Broadcast frame to all connected clients"""
        try:
            # Convert frame to bytes
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = base64.b64encode(buffer).decode('utf-8')
            
            # Create message
            message = {
                'type': 'frame',
                'data': frame_bytes
            }
            
            # Encrypt and send to all clients
            encrypted_message = self.cipher_suite.encrypt(
                json.dumps(message).encode('utf-8')
            )
            
            for client in self.clients[:]:  # Use a copy of the list
                try:
                    client.send(encrypted_message)
                except:
                    self.clients.remove(client)
                    
        except Exception as e:
            self.logger.error(f"Error broadcasting frame: {str(e)}")
            
    def stop(self):
        """Stop the security server"""
        self.is_running = False
        if self.server_socket:
            self.server_socket.close()
        for client in self.clients:
            client.close()
        self.clients.clear()
        self.logger.info("Server stopped") 