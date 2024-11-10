from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import gdown

# Initialize Flask app
app = Flask(__name__)

# Function to download model from Google Drive
def download_model_from_drive(model_url, model_path):
    try:
        # Use gdown to download the model from the provided Google Drive link
        gdown.download(model_url, model_path, quiet=False)
    except Exception as e:
        print(f"Error downloading the model: {e}")

# Google Drive link to your model
model_url = 'https://drive.google.com/uc?id=1Vgtrw1Lf7KfLO-sbB8Iytfiyz9J5DTRR'  # Use the file ID from your shared link

# Path to save the downloaded model
model_path = 'rice_type_classification_with_fluid_mechanics.h5'

# Download model if it doesn't exist
if not os.path.exists(model_path):
    download_model_from_drive(model_url, model_path)

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Class mapping for predictions
class_mapping = {0: 'arborio', 1: 'basmati', 2: 'ipsala', 3: 'jasmine', 4: 'karacadag'}

# Function to preprocess input image
def preprocess_image(image):
    try:
        img = load_img(image, target_size=(224, 224))  # Resize image to 224x224
        img_array = img_to_array(img) / 255.0  # Normalize image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get the uploaded image and fluid behavior input from form
            image = request.files['image']
            fluid_behavior = int(request.form['fluid_behavior'])

            if not image:
                return render_template('index.html', error="Please upload an image.")

            # Save uploaded image temporarily
            upload_folder = 'uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            image_path = os.path.join(upload_folder, image.filename)
            image.save(image_path)

            # Preprocess the image
            input_image = preprocess_image(image_path)

            if input_image is None:
                return render_template('index.html', error="Failed to preprocess the image.")

            # Prepare fluid behavior as input (reshape to match expected input)
            fluid_behavior_input = np.array([[fluid_behavior]])

            # Predict using the model
            prediction = model.predict([input_image, fluid_behavior_input])
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = class_mapping[predicted_class]

            # Clean up saved image
            os.remove(image_path)

            return render_template('index.html', prediction=predicted_label)

        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('index.html', error="An error occurred during prediction.")
    
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
