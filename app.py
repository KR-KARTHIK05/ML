from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import librosa
import numpy as np
import joblib
import traceback
from pydub import AudioSegment

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = joblib.load('models/rf_1_opt.pkl')  # ✅ Load correct model

# Label names in the same order used during model training
label_names = ['calm', 'happy', 'sad', 'angry']

# Emotion mapping as per your logic
emotion_mapping = {
    "angry": "sad",
    "sad": "sad",
    "happy": "calm",
    "calm": "angry"
}

# --- Feature Extraction Functions ---
def ext_mfcc(path):
    y, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])  # 26

def ext_spec_cent(path):
    y, sr = librosa.load(path, sr=None)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.array([np.mean(centroid), np.std(centroid)])  # 2

def ext_spec_roll(path, roll_percent=0.85):
    y, sr = librosa.load(path, sr=None)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=roll_percent)
    return np.array([np.mean(rolloff), np.std(rolloff)])  # 2

def ext_zcr(path):
    y, sr = librosa.load(path, sr=None)
    zcr = librosa.feature.zero_crossing_rate(y)
    return np.array([np.mean(zcr), np.std(zcr)])  # 2

def ext_spec_bd(path):
    y, sr = librosa.load(path, sr=None)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return np.array([np.mean(spec_bw), np.std(spec_bw)])  # 2

def ext_spec_con(path):
    y, sr = librosa.load(path, sr=None)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return np.concatenate([np.mean(contrast, axis=1), np.std(contrast, axis=1)])  # 14

def ext_chroma(path):
    y, sr = librosa.load(path, sr=None)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.concatenate([np.mean(chroma, axis=1), np.std(chroma, axis=1)])  # 24

def ext_rms(path):
    y, sr = librosa.load(path, sr=None)
    rms = librosa.feature.rms(y=y)
    return np.array([np.mean(rms), np.std(rms)])  # 2

def extract_all_features(path):
    f1 = ext_mfcc(path)
    f2 = ext_spec_cent(path)
    f3 = ext_spec_roll(path)
    f4 = ext_zcr(path)
    f5 = ext_spec_bd(path)
    f6 = ext_spec_con(path)
    f7 = ext_chroma(path)
    f8 = ext_rms(path)
    features = np.concatenate([f1, f2, f3, f4, f5, f6, f7, f8])  # 74 total
    return features.reshape(1, -1)

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')  # Ensure this file exists in templates/

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        # Convert to WAV
        wav_filename = os.path.splitext(filename)[0] + '.wav'
        wav_path = os.path.join(app.config['UPLOAD_FOLDER'], wav_filename)

        # Auto-detect format from extension
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)  # Optional: standardize format
        audio.export(wav_path, format='wav')

        # Extract features from WAV file
        features = extract_all_features(wav_path)
        print("📊 Feature shape:", features.shape)

        pred = model.predict(features)
        predicted_index = int(pred[0])
        original_emotion = label_names[predicted_index]
        mapped_emotion = emotion_mapping.get(original_emotion, original_emotion)

        print(f"🧠 Model prediction: {original_emotion} → Mapped: {mapped_emotion}")
        return jsonify({'emotion': mapped_emotion})

    except Exception as e:
        print("🔥 ERROR during prediction:")
        traceback.print_exc()
        return jsonify({'error': 'Prediction failed. Check Flask server.'}), 500
    
# --- Main ---
if __name__ == '__main__':
    app.run(debug=True)
