from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_model("mnist_cnn.h5")

ALLOWED = {'png', 'jpg', 'jpeg', 'bmp'}


def allowed_file(name):
    return '.' in name and name.rsplit('.', 1)[1].lower() in ALLOWED


def preprocess_image(path):
    # 1) Load grayscale
    img = Image.open(path).convert('L')

    # 2) Make square by padding (preserve aspect ratio)
    w, h = img.size
    sz = max(w, h)
    padded = Image.new('L', (sz, sz), color=255)  # white background
    padded.paste(img, ((sz - w) // 2, (sz - h) // 2))

    # 3) Resize to 28x28
    small = padded.resize((28, 28), Image.LANCZOS)

    # 4) Convert to numpy array
    arr = np.array(small).astype(np.float32)

    # If background is light, invert to MNIST style (white digit on black)
    if arr.mean() > 127:
        arr = 255.0 - arr

    # 5) Normalize and reshape
    arr = arr / 255.0
    arr = arr.reshape(1, 28, 28, 1)

    return arr


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify(error="no file"), 400

    f = request.files['file']

    if not (f and allowed_file(f.filename)):
        return jsonify(error="bad file type"), 400

    filename = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(path)

    X = preprocess_image(path)
    probs = model.predict(X)[0]

    pred = int(np.argmax(probs))
    conf = float(np.max(probs))

    return jsonify(prediction=pred, confidence=conf, filename=filename)


if __name__ == '__main__':
    app.run(debug=True)
