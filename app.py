# Import Libraries
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models

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


### Define Generator

## Settings
latent_dim = 128
num_classes = 10

# define layers
## label input
in_label = layers.Input(shape=(1,))
label_embedding = layers.Embedding(num_classes, 50)(in_label)
# scale up to low resolution image dimensions
n_nodes = 7 * 7
label_x = layers.Dense(n_nodes)(label_embedding)
label_x = layers.Reshape((7, 7, 1))(label_x)

## image input
in_lat_vec = layers.Input(shape=(latent_dim,))

x = layers.Dense(7 * 7 * 128)(in_lat_vec)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Reshape((7, 7, 128))(x)

# concatenate images and labels
concat = layers.Concatenate()([x, label_x])

x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(concat)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.BatchNormalization()(x)

out = layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid")(x)

# define model
generator = models.Model(inputs=[in_lat_vec, in_label], outputs=out)

# dummy prediction for initializing generator
generator([np.zeros((1, latent_dim)), np.zeros((1, 1))], training=False)

## Load Generator Model
epoch = 50
model_name = "cgan_1"
generator.load_weights(os.path.join("model", model_name, "gen_epoch{}".format(epoch), "gen"))


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