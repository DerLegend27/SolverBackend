from http.server import BaseHTTPRequestHandler,HTTPServer
import io
import PIL.Image as Image
import cgi
import base64
import json
import logging
import cv2
import time

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
            img.save("/Users/nicolai/Documents/SolverBackend/images/math-equation.png")

            json_string = json.dumps({
                'status': 'successful',
                'solution': 'Hier steht eine Lösung!'
            })

            self._set_headers()
            self.wfile.write(json_string.encode(encoding='utf-8'))

            processing_image()

            return

        except IOError:
            self.send_error(415, "file-extension is not allowed")

def processing_image():
    # Read Input image
    inputImage = cv2.imread("/Users/nicolai/Documents/SolverBackend/images/math-equation.png")

    # Copy image
    inputImageCopy = inputImage.copy()

    # Convert BGR to grayscale
    grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

    # Threshold
    binaryImage = cv2.adaptiveThreshold(grayscaleImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 85, 10)

    # Flood-fill
    cv2.floodFill(binaryImage, None, (0, 0), 0)

    # Find contours
    contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Cropping
    cropped_images = 0

    # Look for the boxes
    for _, c in enumerate(contours):

        # Bounding rectangle
        boundRect = cv2.boundingRect(c)

        # Get rectangle data
        rectX = boundRect[0]
        rectY = boundRect[1]
        rectWidth = boundRect[2]
        rectHeight = boundRect[3]

        # Estimate rectangle area
        rectArea = rectWidth * rectHeight

        # Minimum Area Threshold
        minArea = 15

        # Filter blobs by area:
        if rectArea > minArea:

            cropped_images += 1

            # Draw bounding box:
            color = (0, 255, 0)
            cv2.rectangle(inputImageCopy, (int(rectX), int(rectY)),
                        (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2)

            # Crop bounding box:
            currentCrop = inputImage[rectY:rectY+rectHeight,rectX:rectX+rectWidth]
            cv2.imwrite("/Users/nicolai/Documents/SolverBackend/images/cropped_" + str(cropped_images) + ".png", currentCrop)

    return


def runserver():
    try:
        print("HTTP-Server is starting...")
        http_server = HTTPServer((HOST, PORT), RequestHandler)
        print("HTTP-Server is running...")

        http_server.serve_forever()

    except KeyboardInterrupt:
        http_server.server_close()

runserver()