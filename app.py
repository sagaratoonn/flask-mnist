from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from flask import Flask
import base64
from io import BytesIO


model = None
imagesND = None
im = None
app = Flask(__name__)
 
@app.route('/')
def hello():
    hello = "Detected"
    global imagesND
    global model

    data = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAxCAIAAAA0kjydAAADkUlEQVR4nNyYP0v7ahTHe0uINgQJRURKcSiioQTnIqWziEiR4AuQIiIdxEkcRBxKX0AR6SBSSnBwLFIcJIQMRaSIiNTiEIqEDBJKCKEGMXco/EhzTtLoxeH+zubD+X4/j8+fc56GcBwn8psR/VX3vwJAjM14eXlpNBqiKHa73ff3d8MwGIaJx+Msy+ZyubW1tcXFxSC94x9PT0/r6+vRaNB/GY1G8/n88/Ozn4kv4Pz8nKKokOtAUVStVvsGoFqthrR2x8XFRShAq9UiSfIHAJIk7+7uxgOy2ewP3IeRy+XGAGRZRpUURZVKJUVRbNtWFKVcLtM0jWbKshwEODg4gBqCIG5vbz2ZkiQRBHLKDw8PgwCZTAZqCoUCXEnHcXZ2dmDy8vJyECCZTEKNJEkoQJIkmDw3NxcEmJychJp+v48CdF2HyRRFuXNC1SK//ZyamoKDX19f7j+9gOnpaagxDAMF9Pt9ODgzMxMEWFpagprHx0cUgI57HLyA1dVVqKnX6yhAEAQ46HWA+8YwjEdDEIQoip5MURThPWAYRtf1oFPkOE6lUoHzomm6XC73er0/Nxmttaenpx43vJoWCgV0TYIDvY++/aBUKqF3Ag2SJE9OTlCfoI72+vq6tbU11p3n+U6n42fiC9A0rVgswg2HQdP09va2qqrfAFxeXvrd3gCMIAihAOgpChmVSmUMQBRF9BnBcZwgCKqq2ratqmq9Xuc4DqYRBOEpvSOAz89PlmWhjOd527Y9UxkMBhsbGzA5nU77AhqNBhTMz89bloVulWVZqVQKSq6vr//kjKzG1dUVzN7b24vFYnA8EonEYrH9/X04PuLjnlE6nYbZ3W4Xnf4wOp0OlHAchy8R2kAGg0EAwLIsKGEYBl8iNNvTocKE2yfqIcPsXq8X4KUoChx0+4wA0H55c3MTAGg2m3BwxMe9oDzPw+xUKmWaJroBhmGgz5zNzU18k8/OztBp5vN5uNWWZaH9NRKJVKtVHPD29ub3rh6WCk3ThqWiVquhZ3rYG9yV1VuLdnd3UVn4KBaLbkMvQNO0RCLxY/dkMqlpWhDAcZz7+/vwP57cQdN0u932uOENp91uLywsfMudZdmHhwdo5dsyTdM8OjryvAPRmJ2dPT4+9jvK/wR/q/j4+Gg2m7Ist1otVVV1XTdNk6bpeDyeSCQymUw2m11ZWZmYmPBzGAP47/H//1bx64B/AwAA//8hAgL3rTY0NQAAAABJRU5ErkJggg=="

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
