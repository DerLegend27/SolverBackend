from http.server import BaseHTTPRequestHandler,HTTPServer
import io
import cgi

HOST = "127.0.0.1"
PORT = 8080

class RequestHandler(BaseHTTPRequestHandler):

    def _set_response(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers

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
            

    def do_POST(self):
        
        ctype, pdict = cgi.parse_header(self.headers["Content-Type"])
        
        if ctype == "multipart/form-data":
            pdict["boundary"] = bytes(pdict["boundary"], "utf-8")
            form = cgi.parse_multipart(self.rfile, pdict)

        self._set_response()
        print(form.get("data"))


if 1+1==2:
    print("[Server] wird gestartet...")
    http_server = HTTPServer((HOST, PORT), RequestHandler)
    print("[Server] ist gestartet...")

    http_server.serve_forever()