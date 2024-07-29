import os
import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify
from trafilatura import fetch_url, extract
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load the trained model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Extract content from HTML using Trafilatura
def extract_content(html_content):

    text_content = extract(html_content)
    return text_content

# Generate embedding using Sentence Transformer
def generate_embedding(text_content, model_name):
    model = SentenceTransformer(model_name)
    embedding = model.encode([text_content])
    return embedding

# Load the best-performing model
model_path = "model/xgboost_model_roberta.pkl"  # Adjust the path accordingly

best_model = pickle.load(open(model_path, "rb"))
@app.route("/")
def main_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the POST request has a file part
        if 'html_file' not in request.files:
            return jsonify({'error': 'No file part'})

        html_file = request.files['html_file']

        # Check if the file is an HTML file
        if html_file.filename == '' or not html_file.filename.endswith('.html'):
            return jsonify({'error': 'Invalid file format. Please upload an HTML file'})

        # Save the file
        file_path = os.path.join('test', "page.html")
        html_file.save(file_path)

        # Read the content from the HTML file
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        # Extract content from HTML using Trafilatura
        text_content = extract_content(html_content)

        # Generate embedding using Sentence Transformer
        embedding = generate_embedding(text_content, "aditeyabaral/sentencetransformer-xlm-roberta-base")  # Adjust the model name accordingly
        embedding_2d = np.vstack(embedding)
        # Predict using the trained model

        prediction = best_model.predict(embedding_2d)[0]
        if prediction:
            pred = "html is bening"
        else:
            pred = "html is Phishing"

        # Return the prediction
        return render_template('index.html', prediction_result=f'Prediction: {pred}')

    except Exception as e:
        return render_template('index.html', prediction_result=f'Error: {str(e)}')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=True)
