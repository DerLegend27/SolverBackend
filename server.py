import os
from flask import Flask, request
from PIL import Image
import json
import base64
from flask_ngrok import run_with_ngrok
from flask_cors import CORS
from pre_processing import processing_image
from classifier import classification

app = Flask(__name__)
CORS(app)

# --> Ngrok Application for debug purpose only <--
#run_with_ngrok(app)

@app.route('/api', methods=['POST'])
def handle_form():
    #print("Posted file: {}".format(request.form['image']))

    image = request.form['image']
    #image = request.files["image"]

    #img = Image.open(image)  
    #img.save("test-images/math-equation.png")
    
    # Save image
    with open("test-images/math-equation.png", "wb") as fh:
        fh.write(base64.b64decode(image))


    processing_image()
    os.remove("test-images/math-equation.png")

    solution = classification()
    #solution = "Lewin ist geil"

    json_string = json.dumps({
        'status': 'successful',
        'solution': solution
    })
    
    return json_string

@app.route("/")
def home():

    json_string = json.dumps({
                'Response': 'Um Ergebnisse zu erhalten, mache bitte eine POST-Request!'
            })
    
    return json_string;   


if __name__ == "__main__":
    app.run()
