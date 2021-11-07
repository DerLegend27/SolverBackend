from http.server import BaseHTTPRequestHandler,HTTPServer
import io
import PIL.Image as Image
import cgi
import base64
import json
import logging
from Solver import processing_image
import tkinter as tk
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('TkAgg')

HOST = "127.0.0.1"
PORT = 8080

parser = argparse.ArgumentParser()
parser.add_argument('--show', type=int)
args = parser.parse_args()

class RequestHandler(BaseHTTPRequestHandler):

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

    def do_GET(self):
        logging.warning("**GET**")
        logging.warning(self.headers)
        
        json_string = json.dumps({
                'Response': 'Um Ergebnisse zu erhalten, mache bitte eine POST-Request!'
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
            img = Image.open(io.BytesIO(b[0]))
            img.save("images/math-equation.png")

            solution = processing_image()

            json_string = json.dumps({
                'status': 'successful',
                'solution': solution
            })

            self._set_headers()
            self.wfile.write(json_string.encode(encoding='utf-8'))

            # For debug purpose only
            if args.show == 1:

                root = tk.Tk()

                mainframe = tk.Frame(root)
                mainframe.pack()
                root.title("LaTex")

                label = tk.Label(mainframe)
                label.pack()

                fig = matplotlib.figure.Figure(figsize=(15, 5), dpi=100)
                ax = fig.add_subplot(111)

                canvas = FigureCanvasTkAgg(fig, master=label)
                canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
                canvas._tkcanvas.pack(side="top", fill="both", expand=True)

                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                graph(canvas, ax, solution)

                root.bind("<Return>", graph)
                root.mainloop()

            os.remove(r"C:\Users\Schlu\Documents\Programmieren\SolverBackend\images\math-equation.png")
            os.remove(r"C:\Users\Schlu\Documents\Programmieren\SolverBackend\images\processed-image.bmp")

            return

        except IOError:
            self.send_error(415, "file-extension is not allowed")

def graph(canvas, ax, solution, event=None):
    tmptext = "$"+solution+"$"

    ax.clear()
    ax.text(0.2, 0.6, tmptext, fontsize=50)  
    canvas.draw()

def runserver():
    try:
        print("HTTP-Server is starting...")
        http_server = HTTPServer((HOST, PORT), RequestHandler)
        print("HTTP-Server is running...")

        http_server.serve_forever()

    except KeyboardInterrupt:
        http_server.server_close()

runserver()