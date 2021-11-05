from http.server import BaseHTTPRequestHandler,HTTPServer
import io
import PIL.Image as Image
import cgi
import base64
import json
import logging
import cv2
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
            img = Image.open(io.BytesIO(b[0]))
            img.save("/Users/nicolai/Documents/SolverBackend/images/math-equation.png")

            json_string = json.dumps({
                'status': 'successful',
                'solution': 'Hier steht eine Lösung!'
            })

            self._set_headers()
            self.wfile.write(json_string.encode(encoding='utf-8'))

            processing_image1()

            return

        except IOError:
            self.send_error(415, "file-extension is not allowed")

def processing_image1():
    # Read Input image
    inputImage = cv2.imread("/Users/nicolai/Documents/SolverBackend/images/math-equation.png")

    # Copy image
    inputImageCopy = inputImage.copy()

    # Convert BGR to grayscale
    grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

    # Threshold
    binaryImage = cv2.adaptiveThreshold(grayscaleImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 85, 10)

    # MorphologyEx
    morph = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 27)))

    #cv2.imshow("binary", binaryImage)
    #cv2.waitKey(0)

    # Flood-fill
    #cv2.floodFill(binaryImage, None, (0, 0), 0)

    # Find contours
    contours, h = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Save Array
    conturs_list = list()

    boundary=[]
    for c,cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        boundary.append((x,y,w,h))
    
    c = sorted(boundary, key=lambda b:b[0], reverse=False)

    # Look for the boxes
    for i in range(len(c)):

        coords = c[i]

        # Bounding rectangle
        #boundRect = cv2.boundingRect(i)

        #print(boundRect)

        # Get rectangle data
        rectX = coords[0]
        rectY = coords[1]
        rectWidth = coords[2]
        rectHeight = coords[3]

        conturs_list.append((i, rectX, rectY)) 

        # Estimate rectangle area
        rectArea = rectWidth * rectHeight

        # Minimum Area Threshold
        minArea = 15

        # Filter blobs by area:
        if rectArea > minArea:

            # Draw bounding box:
            color = (0, 255, 0)

            cv2.rectangle(inputImageCopy, (int(rectX), int(rectY)),
                        (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2)
            
            cv2.putText(inputImageCopy, str(i), (int(rectX), int(rectY)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)  

            # Crop bounding box:
            currentCrop = inputImage[rectY:rectY+rectHeight,rectX:rectX+rectWidth]
            cv2.imwrite("/Users/nicolai/Documents/SolverBackend/images/cropped_" + str(i) + ".png", currentCrop)
            
            #cv2.imshow(str(i), currentCrop)
            #cv2.waitKey(0)


    cv2.imshow("test", inputImageCopy)
    print(conturs_list)
    print(conturs_list[2][1])
    cv2.imwrite("/Users/nicolai/Documents/SolverBackend/images/Finalstep1.png", inputImageCopy)
    cv2.waitKey(0)

    processing_image2(c, conturs_list)


def processing_image2(c, conturs_list):
    
    for i in range(len(c)):
        # Read Input image
        inputImage = cv2.imread("/Users/nicolai/Documents/SolverBackend/images/cropped_" + str(i) + ".png")

        # Copy image
        inputImageCopy = cv2.imread("/Users/nicolai/Documents/SolverBackend/images/Finalstep1.png")

        # Convert BGR to grayscale
        grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

        # Threshold
        binaryImage = cv2.adaptiveThreshold(grayscaleImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 85, 10)

        # Find contours
        contours, h = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        boundary=[]
        for c,cnt in enumerate(contours):
            x,y,w,h = cv2.boundingRect(cnt)
            boundary.append((x,y,w,h))
        
        c = sorted(boundary, key=lambda b:b[1], reverse=False)

        if len(c) <= 1:
            continue
        
        print(len(c))

        # Look for the boxes
        for x in range(len(c)):

            coords = c[x]

            # Get rectangle data
            rectX = coords[0]
            rectY = coords[1]
            rectWidth = coords[2]
            rectHeight = coords[3]
                
            # Estimate rectangle area
            rectArea = rectWidth * rectHeight

            # Minimum Area Threshold
            minArea = 40

            # Filter blobs by area:
            if rectArea > minArea:

                # Draw bounding box:
                color = (42, 27, 250)

                cv2.rectangle(inputImageCopy, (int(rectX) + int(conturs_list[i][1]), int(rectY) + int(conturs_list[i][2])),
                            (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2)
                
                cv2.putText(inputImageCopy, str(i), (int(rectX), int(rectY)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (42, 27, 250), 2)  

                # Crop bounding box:
                currentCrop = inputImage[rectY:rectY+rectHeight,rectX:rectX+rectWidth]
                cv2.imwrite("/Users/nicolai/Documents/SolverBackend/images/cropped_" + str(i) + "." + str(x) + ".png", currentCrop)
                
                cv2.imshow(str(i), inputImageCopy)
                cv2.waitKey(0)


    cv2.imshow("test", inputImageCopy)
    cv2.imwrite("/Users/nicolai/Documents/SolverBackend/images/Finalstep2.png", inputImageCopy)
    cv2.waitKey(0)


def runserver():
    try:
        print("HTTP-Server is starting...")
        http_server = HTTPServer((HOST, PORT), RequestHandler)
        print("HTTP-Server is running...")

        http_server.serve_forever()

    except KeyboardInterrupt:
        http_server.server_close()

runserver()