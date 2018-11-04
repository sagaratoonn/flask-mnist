from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from flask import Flask, jsonify, request
# from flasgger import Swagger
import base64
from io import BytesIO

model = None
imagesND = None
im = None

app = Flask(__name__)
# swagger = Swagger(app)
 
@app.route('/recognise', methods=['POST'])
def hello():
    hello = "Detected"
    global imagesND
    global model

    if request.headers['Content-Type'] != 'application/json':
        print(request.headers['Content-Type'])
        return flask.jsonify(res='error'), 400

    print(request.json)
    data = request.json.get('data')

    images = []
    im = Image.open(BytesIO(base64.b64decode(data))).convert('L')
    im = im.resize((28,28))
    im = np.asarray(im, dtype="float32")
    im = im.reshape(28,28)
    im = im / 255.0
    images.append(im)
        
    pd = model.predict_classes(np.array(images, dtype=float))
    return str(pd)
 
if __name__ == "__main__":
    # create Keras DNN Model
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Load Images (0-9 numbers)
    images = []
    for i in range(10):
        image = Image.open('./numbers/pict-' + str(i) + '.png').convert('L')
        image = image.resize((28,28))
        im = np.asarray(image, dtype="float32")
        im = im.reshape(28,28)
        im = im / 255.0
        images.append(im)
        
    imagesND = np.array(images, dtype=float)
    
    # Create Training Labels
    label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    label = np.array(label, dtype=int)
    
    # Fitting
    model.fit(imagesND, label, epochs=5)
    pd = model.predict_classes(imagesND)

    print("fit & prediect is complete")
    print(pd)

    app.run(debug=False, host='0.0.0.0', port=5000)
