import os
from flask import Flask, request
from PIL import Image
import json
import base64
from flask_ngrok import run_with_ngrok
from pre_processing import processing_image
from classifier import classification

app = Flask(__name__)

# --> Ngrok Application for debug purpose only <--
#run_with_ngrok(app)

@app.route('/api', methods=['POST'])
def handle_form():

    print("Posted file: {}".format(request.files['image']))
    image = request.files['image']
    img = Image.open(image) 

    img.save("result-images/math-equation.png")

    processing_image("result-images/math-equation.png")
    os.remove("result-images/math-equation.png")

    solution = classification()

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
