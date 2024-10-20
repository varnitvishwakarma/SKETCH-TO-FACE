import tensorflow as tf
from PIL import Image
import numpy as np
from flask import Flask, request, send_file,render_template

# Correct model path
generator = tf.keras.models.load_model('C:/Users/tusha/OneDrive/Documents/sketch to image/models/generator_model.keras')

def load_and_preprocess_image(file_path):
    img = Image.open(file_path).convert('RGB')
    img = img.resize((128, 128))
    img = np.array(img) / 127.5 - 1.0  # Normalize to [-1, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def generate_image_from_sketch(sketch_image):
    # Generate a random noise vector
    noise = tf.random.normal([1, 100])
    generated_image = generator([noise, sketch_image], training=False)
    generated_image = (generated_image[0] + 1) / 2.0  # Rescale to [0, 1]
    return generated_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if 'sketch' not in request.files:
        return "No file uploaded", 400

    sketch_file = request.files['sketch']
    sketch_img = load_and_preprocess_image(sketch_file)
    generated_img = generate_image_from_sketch(sketch_img)

    # Save generated image to a temporary file
    generated_img = (generated_img * 255).numpy().astype(np.uint8)
    img = Image.fromarray(generated_img)
    img.save('generated_image.png')

    # Send the file back as a response
    return send_file('generated_image.png', mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)
