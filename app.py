from flask import Flask, request, render_template, send_from_directory
from keras.models import load_model
import numpy as np
import tensorflow as tf
import io
from PIL import Image
import base64
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


app = Flask(__name__, static_url_path='/static')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


model = load_model('defect_detection_pp_woven.h5')



@app.route('/metrics', methods=['GET'])
def metrics():
    return render_template("metrics.html")


@app.route('/', methods=['POST', 'GET'])
def index():
    version = int(time.time())
    if request.method == "POST":
        
        ImageFile = request.files.get('image').read()
        image_base64 = base64.b64encode(ImageFile).decode("utf-8")
        
        image = Image.open(io.BytesIO(ImageFile))
        image = image.resize((300, 300))

        
        image = image.convert('L')

        
        image = np.array(image)

        
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)
        
        predictions = model.predict(image)
        
        
        if predictions[0][0] < 0.5:
            prediction = "DEFECT"
        else:
            prediction = "OK"
        
        return render_template('index.html', prediction=prediction, image_base64=image_base64, version=version)
    return render_template("index.html", version=version)

if __name__ == '__main__':
    app.run(debug=True)
