from flask import Flask, request, jsonify
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.utils import shuffle
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_FILE = "timestamp_svm_model.pkl"

def train_model():
    print("Training new SVM model...")
    positive = ['2023-05-14', '12:34:56', '2023/08/11', '11:59:59', '2000-01-01 00:00:00']
    negative = ['ERROR', 'INFO', 'login', 'user', 'failed', 'connected']
    
    labels = [1]*len(positive) + [0]*len(negative)
    samples = positive + negative

    vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    X = vectorizer.fit_transform(samples)
    X, labels = shuffle(X, labels, random_state=42)

    model = SVC(probability=True)
    model.fit(X, labels)

    joblib.dump((model, vectorizer), MODEL_FILE)
    print("Model trained and saved.")
    return model, vectorizer

# Load or train the model
try:
    if os.path.exists(MODEL_FILE):
        model, vectorizer = joblib.load(MODEL_FILE)
        print("Loaded existing model.")
    else:
        model, vectorizer = train_model()
except Exception as e:
    print(f"Failed to load model: {e}")
    model, vectorizer = train_model()

def extract_tokens(lines):
    tokens = set()
    for line in lines:
        tokens.update(line.strip().split())
    return list(tokens)

def extract_timestamp_info_from_lines(lines):
    tokens = extract_tokens(lines)
    X_tokens = vectorizer.transform(tokens)
    probs = model.predict_proba(X_tokens)[:, 1]

    timestamp_tokens = [t for t, p in zip(tokens, probs) if p > 0.5]

    parsed_logs = []
    for line in lines:
        timestamp_found = None
        for token in timestamp_tokens:
            if token in line:
                timestamp_found = token
                break
        parsed_logs.append({
            "timestamp": timestamp_found or "N/A",
            "message": line.strip()
        })

    return parsed_logs

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'logfile' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['logfile']
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    with open(file_path, "r", errors='ignore') as f:
        lines = f.readlines()

    parsed_logs = extract_timestamp_info_from_lines(lines)
    return jsonify(parsed_logs)

@app.route('/send_logs', methods=['POST'])
def send_logs():
    data = request.get_json()
    if not data or 'logs' not in data:
        return jsonify({"error": "No logs provided"}), 400

    # Expecting a single string with multiple lines
    log_content = data['logs']
    if isinstance(log_content, list):
        lines = log_content
    else:
        lines = log_content.strip().split('\n')

    parsed_logs = extract_timestamp_info_from_lines(lines)
    return jsonify(parsed_logs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
