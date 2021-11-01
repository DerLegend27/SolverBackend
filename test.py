from http.server import BaseHTTPRequestHandler,HTTPServer
import time

HOST = "127.0.0.1"
PORT = 80

class RequestHandler(BaseHTTPRequestHandler):

    def post(self):
        pass


if __name__ == "__main__ ":
    print("[Starting the server]")
    
    http_server = HTTPServer((HOST, PORT), RequestHandlerClass)