from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
from PIL import Image
import numpy as np
import io
from PIL import Image
from flask import Flask, render_template, request
from werkzeug.exceptions import BadRequest
from inference import detect_face

app = Flask(__name__)

@app.route('/', methods=['GET'])
def get():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def predict():
    file = extract_img(request)
    image = file.read()
    image = Image.open(io.BytesIO(image))
    image = np.array(image)
    detect_face(image) 
    return render_template("index.html",)

# extract the image from the request
def extract_img(request):
    if 'file' not in request.files:
        raise BadRequest("Missing file parameter!")
        
    file = request.files['file']
    
    if file.filename == '':
        raise BadRequest("Given file is invalid")
    
    return file
    
if __name__ == "__main__":
	print("App run!")

	app.run(debug=True)