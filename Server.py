from http.server import BaseHTTPRequestHandler,HTTPServer
import io
import PIL.Image as Image
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
            self.wfile.write(html.read())
            html.close()

            self._set_response()

        except IOError:
            self.send_error(404, "file not found")
            

    def do_POST(self):
        
        ctype, pdict = cgi.parse_header(self.headers["Content-Type"])

        if ctype == "multipart/form-data":
            pdict["boundary"] = bytes(pdict["boundary"], "utf-8")
            form = cgi.parse_multipart(self.rfile, pdict)
        
        b = form.get("image")
        
        try:
            img = Image.open(io.BytesIO(b[0]))
            img.show()

            self._set_response()

        except IOError:
            self.send_error(415, "file-extension is not allowed")

def run():
    print("HTTP-Server is starting...")
    http_server = HTTPServer((HOST, PORT), RequestHandler)
    print("HTTP-Server is running...")

    http_server.serve_forever()


if __name__ == '__main__':
    run()