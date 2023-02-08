from flask import Flask, request, render_template
from keras.models import load_model
import numpy as np
import tensorflow as tf
import io
from PIL import Image
import cv2


app = Flask(__name__)

# Load the model
model = load_model('defect_detection_pp_woven.h5')

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        # Get the image from the request
        ImageFile = request.files.get('image').read()
        
        # Resize the image to 300x300
        image = Image.open(io.BytesIO(ImageFile))
        image = image.resize((300, 300))

        # Convert the image to grayscale
        image = image.convert('L')

        # Convert the image to a numpy array
        image = np.array(image)

        # Add a batch dimension
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)
        # Make predictions with the model
        predictions = model.predict(image)

        # prediction = str(predictions)
        # Check if the prediction is "def" or "ok"
        if predictions[0][0] < 0.5:
            prediction = "Given Image is defect"
        else:
            prediction = "Given Image is ok"
        
        return render_template('index.html', prediction=prediction)
    return render_template("index.html")

if __name__ == '__main__':
    app.run()
