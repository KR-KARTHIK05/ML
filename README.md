# Speech Emotion Recognition System

This project is a comprehensive pipeline for building a Speech Emotion Recognition (SER) system. It integrates multiple public emotion-labeled speech datasets, performs extensive feature extraction, trains various machine learning models (including Optuna-tuned classifiers), and deploys a Flask-based web server for real-time emotion prediction via audio upload.

---

## Table of Contents

* [Overview](#overview)
* [Datasets](#datasets)
* [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
* [Modeling](#modeling)

  * [Trained Models](#trained-models)
  * [Evaluation](#evaluation)
* [Web Interface](#web-interface)
* [Setup Instructions](#setup-instructions)
* [Directory Structure](#directory-structure)
* [Future Work](#future-work)
* [License](#license)

---

## Overview

The system classifies speech samples into four emotion categories:

* **Angry**
* **Calm**
* **Happy**
* **Sad**

It uses a wide range of audio features such as MFCC, Spectral Centroid, Chroma, and more. These are extracted using `librosa` and used to train classifiers like:

* Support Vector Machine (SVM)
* Decision Tree
* Random Forest
* XGBoost
* AdaBoost

The final deployed model uses **Random Forest with Optuna hyperparameter tuning** and achieves high accuracy with MFCC features.

---

## Datasets

The project uses four benchmark datasets for training and evaluation:

1. **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song
2. **CREMA-D**: Crowd-sourced Emotional Multimodal Actors Dataset
3. **SAVEE**: Surrey Audio-Visual Expressed Emotion
4. **TESS**: Toronto Emotional Speech Set

Classes like *disgust*, *fear*, and *surprise* were excluded to focus on the four key emotions.

---

## Preprocessing & Feature Engineering

### Data Preparation

* Audio files were labeled by parsing filenames and directory structures.
* Stratified splitting ensured balanced representation of each class in training/testing.

### Features Extracted (Total: 74 features)

| Feature Type       | Description                            | Dimensionality |
| ------------------ | -------------------------------------- | -------------- |
| MFCC               | Mel-Frequency Cepstral Coefficients    | 26             |
| Spectral Centroid  | Brightness of the sound                | 2              |
| Spectral Rolloff   | Frequency below which most energy lies | 2              |
| Zero Crossing Rate | Number of zero-crossings in signal     | 2              |
| Spectral Bandwidth | Spread of frequencies                  | 2              |
| Spectral Contrast  | Difference in spectral peaks/valleys   | 14             |
| Chroma             | Energy in 12 pitch classes             | 24             |
| RMS Energy         | Root mean square energy                | 2              |

---

## Modeling

### Trained Models

Each feature set was evaluated using multiple classifiers:

* **Support Vector Machine**
* **Decision Tree**
* **Random Forest**
* **XGBoost**
* **AdaBoost**

Both default models and **Optuna-tuned** versions were trained.

### Evaluation

| Feature           | Best Model (Accuracy %)         |
| ----------------- | ------------------------------- |
| MFCC              | XGBoost (Optuna): 76.78%        |
| Spectral Centroid | XGBoost (Optuna): 44.03%        |
| Spectral Rolloff  | XGBoost (Optuna): 38.88%        |
| ZCR               | Random Forest (Optuna): \~41.4% |
| Others            | Lower performance than MFCC     |

MFCC features consistently outperformed others and were selected for deployment.

---

## Web Interface

A Flask-based API is developed to allow real-time emotion prediction.

### Endpoints

* `/` – Serves the HTML upload form.
* `/predict` – Accepts an audio file (any common format), extracts features, and returns predicted emotion in JSON.

### How It Works

1. User uploads an audio file.
2. The system standardizes the format (mono, 16kHz).
3. Features are extracted and passed to the trained model.
4. The predicted emotion is returned to the user.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/KR-KARTHIK05/ML.git
cd ML
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

Make sure to also install `ffmpeg` for `pydub` audio conversion to work.

### 4. Run the Web App

```bash
python app.py
```

Visit `http://localhost:5000` to upload audio and test the system.

---

## Directory Structure

```text
ML/
├── app.py                     # Flask server
├── models/
│   └── rf_1_opt.pkl           # Final trained model
├── uploads/                   # Stores uploaded audio files
├── templates/
│   └── index.html             # Frontend UI
```

---

## Future Work

* Add support for real-time microphone input.
* Expand emotion classes to include fear, disgust, surprise.
* Train deep learning models (e.g., CNN, RNN).
* Improve frontend UI with charts and audio waveform.
* Integrate music recommendation based on detected emotion.

---

## License

This project is licensed under the [MIT License](LICENSE).