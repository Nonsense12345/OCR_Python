from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image
import ocr as ocr

app = Flask(__name__)

@app.route('/extract-text', methods=['POST'])
def extract_text_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file_storage = request.files['image']
    img_bytes = file_storage.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    text = ocr.extract_text_from_image(img)
    return jsonify({'text': text})

if __name__ == '__main__':
    app.run(debug=True)