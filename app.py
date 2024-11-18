import streamlit as st
import numpy as np
import librosa
import pickle

st.title("Sound Based Deepfake Detectors")
st.text("A security application for social media and web platforms to identify the sound based deepfakes using signal processing infused deep learning framework")

def extract_features(file_path):
    """Extract audio features: MFCC, Chroma, Mel Spectrogram, and Spectral Contrast."""
    y, sr = librosa.load(file_path, duration=5.0, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    mel = librosa.feature.melspectrogram(y=y, sr=sr).mean(axis=1)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)
    return np.hstack([mfccs, chroma, mel, contrast])

audio_file = st.file_uploader("Upload an audio sample", type = [".mp3", ".wav"])

if audio_file is not None:
    with st.spinner("Extracting features and predicting..."):
        audio_features = extract_features(audio_file)
        audio_features = audio_features.reshape(1, -1, 1)

        model = pickle.load(open("best_model.pkl", 'rb'))

        prediction = model.predict(audio_features)

    if prediction > 0.5:
        st.error(f"The audio is Fake with a confidence score: {prediction[0]}")
    else:
        st.success(f"The audio is Real with a confidence score: {1- prediction[0]}")
