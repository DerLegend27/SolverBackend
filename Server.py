import os
from flask import Flask, request, render_template
from PIL import Image
import json
import base64
from flask_ngrok import run_with_ngrok
from Solver import processing_image, calculate

app = Flask(__name__)

run_with_ngrok(app)

@app.route('/api', methods=['POST'])
def handle_form():

    print("Posted file: {}".format(request.files['image']))
    image = request.files['image']
    print(type(image))
    img = Image.open(image) 
    new_img = img.resize((256, 256))

    new_img.save("images/math-equation.png")
    print(new_img.size)

    solution = processing_image()

    json_string = json.dumps({
        'status': 'successful',
        'solution': solution
    })

    os.remove("images/math-equation.png")
    os.remove("images/processed-image.bmp")
    
    return json_string

@app.route('/api/calculate', methods=['POST'])
def handle_formel():

    print("Posted file: {}".format(request.files['solution']))
    equation = request.files['solution']
    
    final = calculate(equation)

    json_string = json.dumps({
        'status': 'successful',
        'solution': final
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
