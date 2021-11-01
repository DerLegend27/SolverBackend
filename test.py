from http.server import BaseHTTPRequestHandler,HTTPServer

HOST = "127.0.0.1"
PORT = 8080

class RequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        
        try:
            html = open("index.html", "rb")

            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            self.wfile.write(html.read())

            html.close()
            return

        except IOError:
            self.send_error(404, "Piss dich!")


if __name__ == "__main__ ":
    print("[Server] wird gestartet...")
    http_server = HTTPServer((HOST, PORT), RequestHandler)
    print("[Server] ist gestartet...")

    http_server.serve_forever()