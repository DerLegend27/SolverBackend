from http.server import BaseHTTPRequestHandler,HTTPServer
import io
import PIL.Image as Image
import cgi
import base64
import json
import logging
import time
import numpy as np
from numpy.core.fromnumeric import size

HOST = "127.0.0.1"
PORT = 8080

class RequestHandler(BaseHTTPRequestHandler):

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

    def do_GET(self):
        logging.warning("**GET**")
        logging.warning(self.headers)
        
        json_string = json.dumps({
                'Hallo': 'DU SOLLST EINE POST-REQUEST MACHEN!'
            })

        self._set_headers()
        self.wfile.write(json_string.encode(encoding='utf-8'))     

    def do_POST(self):
        logging.warning("**POST**")
        logging.warning(self.headers)

        ctype, pdict = cgi.parse_header(self.headers["Content-Type"])

        if ctype != "multipart/form-data":
            self.send_error(400, "is not in a multipart/form-data")
            self.end_headers()
            return
            
        pdict["boundary"] = bytes(pdict["boundary"], "utf-8")
        form = cgi.parse_multipart(self.rfile, pdict)
        
        b = form.get("image")
        
        try:
            img = Image.open(io.BytesIO(base64.b64decode(b[0])))
            img.show()
            img.save("images/math-equation.png")

            json_string = json.dumps({
                'status': 'successful',
                'solution': 'Hier steht eine Lösung!'
            })

            self._set_headers()
            self.wfile.write(json_string.encode(encoding='utf-8'))


            return

        except IOError:
            self.send_error(415, "file-extension is not allowed")


def runserver():
    try:
        print("HTTP-Server is starting...")
        http_server = HTTPServer((HOST, PORT), RequestHandler)
        print("HTTP-Server is running...")

        http_server.serve_forever()

    except KeyboardInterrupt:
        http_server.server_close()

runserver()