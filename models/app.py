import yaml
import os
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Union, ByteString, Any
import flask
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for
import requests
import torch
from torchvision import transforms, models
import warnings
import gdown
import pickle
import google.generativeai as genai

warnings.filterwarnings("ignore")
print("Cowshala: Imports Done")

# Define paths and URLs directly
export_file_url = 'https://drive.google.com/u/0/uc?id=1x5Ljh9xtNfXFMm97AMlewW1nZ77XS-gb&export=download'
export_file_name = 'cattle_breed_classifier_full_model.pth'
path = Path("models/")
export_classes_url = "https://drive.google.com/u/0/uc?id=1IaF_zn-RDnsEntYp86F5G7FNlEkQ8KJ_&export=download"
export_classes_name = "classes.txt"

# Ensure models directory exists
path.mkdir(parents=True, exist_ok=True)

def download_file(url, dest):
    if dest.exists():
        return
    gdown.download(url, str(dest), quiet=False)

def setup_learner():
    download_file(export_file_url, path / export_file_name)
    download_file(export_classes_url, path / export_classes_name)
    try:
        try:
            torch.serialization.load_add_safe_globals({'ResNet': models.resnet.ResNet})
            learn = torch.load(path / export_file_name, map_location=torch.device('cpu'))

        except AttributeError:
            print("Cowshala: Older PyTorch version detected. Using unsafe load.")
            learn = torch.load(path / export_file_name, map_location=torch.device('cpu'), pickle_module=pickle)

        with open(path / export_classes_name, 'r') as file:
            class_list = file.read().split(",")
        return learn, class_list

    except FileNotFoundError:
        print(f"Cowshala: Error: Model file '{export_file_name}' or classes file '{export_classes_name}' not found in 'models/' directory.")
        print("Cowshala: Please download the files from the provided links.")
        sys.exit(1)

model, class_list = setup_learner()

# Load Configuration
with open("config.yaml", 'r') as stream:
    APP_CONFIG = yaml.full_load(stream)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize Gemini API
genai.configure(api_key=APP_CONFIG.get("gemini_api_key"))  # Get Gemini API key from config

def load_image_url(url: str) -> Image:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def load_image_bytes(raw_bytes: ByteString) -> Image:
    img = Image.open(BytesIO(raw_bytes))
    return img

def predict(img, n: int = 3) -> Dict[str, Union[str, List]]:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    outputs = model(transform(img).unsqueeze(0)).squeeze()
    pred_probs = torch.nn.Softmax(dim=-1)(outputs)
    _, pred_class = torch.max(pred_probs, dim=0)
    pred_probs = pred_probs.tolist()
    predictions = []
    for image_class, output, prob in zip(class_list, outputs.tolist(), pred_probs):
        output = round(output, 1)
        prob = round(prob, 2)
        predictions.append(
            {"class": image_class.replace("_", " "), "output": output, "prob": round(prob, 2)}
        )
    predictions = sorted(predictions, key=lambda x: x["output"], reverse=True)
    predictions = predictions[0:n]
    print({"class": str(pred_class.item()), "predictions": predictions})
    return {"class": str(pred_class.item()), "predictions": predictions}

def get_breed_info(breed_name: str) -> List[str]:
    """Uses Gemini to get breed characteristics and regions."""
    model_genai = genai.GenerativeModel('gemini-pro')
    prompt = f"Provide 3-4 key characteristics and 2-3 regions where the {breed_name} breed of cattle is commonly found. Please format the characteristics as a list and the regions as a separate list."
    try:
        response = model_genai.generate_content(prompt)
        print(f"Gemini response: {response.text}") #debugging line
        info = response.text.split('\n')
        return [item.strip() for item in info if item.strip()]
    except Exception as e:
        print(f"Error fetching breed info from Gemini: {e}")
        return ["Characteristics and regions unavailable."]

@app.route('/api/classify', methods=['POST', 'GET'])
def upload_file():
    try:
        if flask.request.method == 'GET':
            url = flask.request.args.get("url")
            img = load_image_url(url)
        else:
            bytes = flask.request.files['file'].read()
            img = load_image_bytes(bytes)
        res = predict(img)
        return flask.jsonify(res)
    except Exception as e:
        return flask.jsonify({"error": str(e)}), 500

@app.route('/api/classes', methods=['GET'])
def classes():
    try:
        with open('models/classes.txt', 'r') as file:
            classes = file.read().split(",")
        return flask.jsonify(classes)
    except FileNotFoundError:
        return flask.jsonify({"error": "Classes file not found."}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return "pong"

@app.route('/config')
def config():
    return flask.jsonify(APP_CONFIG)

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

@app.route('/')
def root():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    if request.method == 'POST':
        if 'image' in request.files and request.files['image'].filename != '':
            image_file = request.files['image']
            img = load_image_bytes(image_file.read())
            res = predict(img)
            predicted_breed = res['predictions'][0]['class'].replace("_", " ").title() #added title and replace
            print(f"Predicted Breed: {predicted_breed}") #debugging line
            breed_info = get_breed_info(predicted_breed)
            return render_template('prediction.html', prediction=predicted_breed, breed_info=breed_info)
        elif 'url' in request.form and request.form['url'] != '':
            image_url = request.form['url']
            img = load_image_url(image_url)
            res = predict(img)
            predicted_breed = res['predictions'][0]['class'].replace("_", " ").title() #added title and replace
            print(f"Predicted Breed: {predicted_breed}") #debugging line
            breed_info = get_breed_info(predicted_breed)
            return render_template('prediction.html', prediction=predicted_breed, breed_info=breed_info)
        else:
            return render_template('prediction.html', prediction="No image provided.", breed_info=[])
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)