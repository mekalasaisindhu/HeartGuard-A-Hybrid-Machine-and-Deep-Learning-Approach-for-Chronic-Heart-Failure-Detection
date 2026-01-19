"""
Simple HTTP server to serve the HTML frontend
Runs on http://localhost:8080
"""
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import sys

class MyHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow requests from frontend
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_GET(self):
        # Serve index.html for root path
        if self.path == '/':
            self.path = '/index.html'
        return super().do_GET()

if __name__ == '__main__':
    # Change to deployment directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    server_address = ('', 8080)
    httpd = HTTPServer(server_address, MyHTTPRequestHandler)
    print("Frontend server started at http://127.0.0.1:8080")
    print("Press Ctrl+C to stop the server")
    httpd.serve_forever()
