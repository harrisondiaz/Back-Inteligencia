import os

import PIL
from flask import Flask, request, jsonify
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

def load_image_from_url(url, size=(128, 128)):
    response = requests.get(url)
    try:
        img = Image.open(BytesIO(response.content))
    except PIL.UnidentifiedImageError:
        raise ValueError("The URL you provided does not point to a valid image.")
    img = img.convert('RGB')
    img = img.resize(size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict(model, img, encoder):
    prediction = np.argmax(model.predict(img), axis=-1)
    return encoder.inverse_transform(prediction)

modelo_guardado = 'modelo_entrenado_excel.h5'
archivo_encoder = 'encoder_classes_excel.npy'

modelo = load_model(modelo_guardado)
encoder_classes = np.load(archivo_encoder, allow_pickle=True)
encoder = LabelEncoder()
encoder.classes_ = encoder_classes

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json(force=True)
    url = data['url']
    img = load_image_from_url(url)
    tipo_hoja = predict(modelo, img, encoder)
    return jsonify({"clase_de_hoja": tipo_hoja[0]})

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=os.getenv("PORT", default=5000))


