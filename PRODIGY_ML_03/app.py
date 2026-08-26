
import os, io, time, base64
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / 'static'
UPLOADS_DIR = STATIC_DIR / 'uploads'
TEMPLATES_DIR = BASE_DIR / 'templates'

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = str(UPLOADS_DIR)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

class CatDogVisionClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.cat_indices = set(range(281, 286))
        self.dog_indices = set(range(151, 269))
        self.model = None
        self._load_model()

    def _load_model(self):
        print('[+] Loading Pre-trained ResNet50 Vision Model...')
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.eval()
        self.model.to(self.device)
        print(f'[+] Model successfully loaded on: {self.device}')

    def predict(self, image: Image.Image) -> Tuple[str, float, Dict[str, float], str]:
        image_rgb = image.convert('RGB')
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy().flatten()
        top5_indices = np.argsort(probabilities)[-5:][::-1]
        cat_prob = float(sum(probabilities[idx] for idx in self.cat_indices))
        dog_prob = float(sum(probabilities[idx] for idx in self.dog_indices))
        top_idx = int(top5_indices[0])
        is_top_cat = top_idx in self.cat_indices
        is_top_dog = top_idx in self.dog_indices
        has_cat_in_top5 = any(idx in self.cat_indices for idx in top5_indices)
        has_dog_in_top5 = any(idx in self.dog_indices for idx in top5_indices)
        if (cat_prob > 0.15 or is_top_cat) and cat_prob > dog_prob and has_cat_in_top5:
            total = cat_prob + dog_prob
            conf = cat_prob / total if total > 0 else cat_prob
            conf = max(conf, cat_prob)
            label = 'Cat'
            msg = f'Detected feline visual features with {conf * 100:.1f}% confidence.'
        elif (dog_prob > 0.15 or is_top_dog) and dog_prob > cat_prob and has_dog_in_top5:
            total = cat_prob + dog_prob
            conf = dog_prob / total if total > 0 else dog_prob
            conf = max(conf, dog_prob)
            label = 'Dog'
            msg = f'Detected canine visual features with {conf * 100:.1f}% confidence.'
        else:
            label = 'Invalid'
            conf = min(1.0 - max(cat_prob, dog_prob), 0.99)
            msg = 'Image does not appear to contain a recognizable cat or dog.'
        other_score = max(0.0, 1.0 - (cat_prob + dog_prob))
        scores = {'Cat': round(cat_prob * 100, 2), 'Dog': round(dog_prob * 100, 2), 'Other': round(other_score * 100, 2)}
        return label, conf, scores, msg

classifier = CatDogVisionClassifier()

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_image_from_request(req):
    if 'image' in req.files:
        f = req.files['image']
        if f and f.filename != '' and allowed_file(f.filename):
            return Image.open(f.stream)
    if 'image_data' in req.form:
        d = req.form['image_data']
        if d.startswith('data:image'):
            d = d.split(',', 1)[1]
        try:
            return Image.open(io.BytesIO(base64.b64decode(d)))
        except Exception:
            pass
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.perf_counter()
        img = extract_image_from_request(request)
        if img is None:
            return jsonify({'success': False, 'error': 'No valid image provided. Please upload a PNG, JPG, JPEG, WEBP, or BMP file.'}), 400
        label, confidence, scores, explanation = classifier.predict(img)
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return jsonify({
            'success': True,
            'label': label,
            'confidence': round(confidence * 100, 2),
            'scores': scores,
            'message': explanation,
            'inference_time_ms': inference_time_ms
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'ResNet-50 (ImageNet-1k)', 'device': str(classifier.device)})

if __name__ == '__main__':
    print('[*] Starting Cat vs Dog Flask Web Application on http://127.0.0.1:5000')
    app.run(host='0.0.0.0', port=5000, debug=False)
