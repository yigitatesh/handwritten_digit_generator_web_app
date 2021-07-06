# Import Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from flask import Flask, request, jsonify, render_template, flash
import os

from PIL import Image
import io
import cv2
import base64


# Create Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Tensorflow Options
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


### Generator

## Settings
latent_dim = 128
num_classes = 10

## Load Generator Model
epoch = 50
model_name = "cgan_1"
file_name = "model/gen_" + model_name + "_epoch" + str(epoch) + ".h5"
generator = load_model(file_name)

# dummy prediction for initializing generator
generator([np.zeros((1, latent_dim)), np.zeros((1, 1))], training=False)


### Some Settings
MAX_DIGIT_NUM = 50


### Generate Digits
def generate_images(num_images=1, class_=0):
    random_vectors = tf.random.normal(shape=(num_images, latent_dim))
    class_vec = tf.ones(shape=(num_images, 1)) * class_
    images = generator([random_vectors, class_vec], training=False)
    return images.numpy()
    
def generate_random_class_images(num_images=1):
    random_vectors = tf.random.normal(shape=(num_images, latent_dim))
    class_vec = tf.ones(shape=(num_images, 1))
    class_vec *= np.random.randint(0, num_classes, num_images).reshape(-1, 1)
    images = generator([random_vectors, class_vec], training=False)
    return images.numpy()

## Helper Functions
def is_digit_type_valid(digit_type):
    if digit_type == "":
        return True
    elif not digit_type.isdigit():
        return False
    return 0 <= int(digit_type) < num_classes


# Flask Functions
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate',methods=['POST'])
def generate():
    '''
    For rendering results on HTML GUI
    '''
    ## preprocess given features
    features = list(request.form.values())
    are_inputs_valid = True

    # process digit class
    digit_type = features[0].strip()

    if not is_digit_type_valid(digit_type):
        flash("Please type a digit (like '5') to generate image(s).")
        are_inputs_valid = False
    else:
        digit_type = None if digit_type == "" else int(digit_type)

    # process number of digits
    num_digits = features[1].strip()

    if num_digits == "":
        num_digits = 1
    else:
        try:
            num_digits = max(int(num_digits), 1)
            # big number warning
            if num_digits > MAX_DIGIT_NUM:
                flash("Please type a number smaller than {} as number of digits".format(MAX_DIGIT_NUM))
                are_inputs_valid = False
        except:
            flash("Please type an integer number as number of digits.")
            are_inputs_valid = False

    # return if inputs are invalid
    if not are_inputs_valid:
        return render_template("index.html")

    # generate digits
    if digit_type == None: # generate random images
        generated_images = generate_random_class_images(num_digits)
    else: # generate digits with a known class
        generated_images = generate_images(num_digits, digit_type)
        
    # Process images to show in html
    images = []
    for img_arr in generated_images:
        data = io.BytesIO()
        img = cv2.resize(img_arr, (84, 84)) # make img a bit bigger
        img = np.uint8(img * 255)
        #img = np.squeeze(img, axis=-1)
        img = Image.fromarray(img, "L") # grayscale img
        img.save(data, "JPEG")
        encoded_img_data = base64.b64encode(data.getvalue())
        img_data = encoded_img_data.decode("utf-8")
        images.append(img_data)

    return render_template('index.html', generated_images=images)


if __name__ == "__main__":
    app.run(debug=True)