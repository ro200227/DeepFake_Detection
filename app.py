import streamlit as st
import numpy as np
import librosa
import pickle
import torch
import torch.nn as nn
import os
import tensorflow as tf

# ----------------------- AUDIO MODEL SETUP -----------------------

class DeepfakeAudioModel(nn.Module):
    def __init__(self, input_size=40):
        super(DeepfakeAudioModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load audio model and label encoder
with open("features.pkl", "rb") as f:
    _, _, _, _, _, _, label_encoder = pickle.load(f)

audio_model = DeepfakeAudioModel(input_size=40)
audio_model.load_state_dict(torch.load("deepfake_audio_model.pth", map_location=torch.device("cpu")))
audio_model.eval()

# Audio feature extraction
def extract_audio_features(file_path, sr=22050, n_mfcc=40):
    audio, sr = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)

# ----------------------- IMAGE MODEL SETUP -----------------------

# Load image classification model
image_model = tf.keras.models.load_model("image.h5")

# Preprocess uploaded image
def preprocess_image(file_path):
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------- STREAMLIT APP -----------------------

st.set_page_config(page_title="Deepfake Detection App", layout="centered")

# Custom background
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
 
    background-size: cover;
}}
[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("🔍 Deepfake Detection System")
st.markdown("Choose the type of media to detect whether it's real or fake.")

option = st.radio("Select Input Type", ["Audio", "Image"], horizontal=True)

# ---------------- AUDIO SECTION ----------------
if option == "Audio":
    uploaded_audio = st.file_uploader("Upload a .wav or .mp3 file", type=["wav", "mp3"])
    if uploaded_audio is not None:
        file_path = "temp_audio.wav"
        # Convert mp3 to wav if needed
        if uploaded_audio.name.endswith(".mp3"):
            from pydub import AudioSegment
            audio = AudioSegment.from_file(uploaded_audio, format="mp3")
            audio.export(file_path, format="wav")
        else:
            with open(file_path, "wb") as f:
                f.write(uploaded_audio.read())

        # Listen to the audio
        st.audio(file_path)

        # Prediction
        features = extract_audio_features(file_path)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = audio_model(features_tensor)
            prediction = torch.argmax(output, dim=1).item()
            label = label_encoder.inverse_transform([prediction])[0]

        st.success(f"Prediction: {label.upper()}")

# ---------------- IMAGE SECTION ----------------
elif option == "Image":
    uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        file_path = os.path.join("temp_image.jpg")
        with open(file_path, "wb") as f:
            f.write(uploaded_image.read())

        st.image(file_path, caption="Uploaded Image",width=300)

        processed_img = preprocess_image(file_path)
        prediction = image_model.predict(processed_img)[0]
        predicted_label = "Fake" if prediction[0] > 0.5 else "Real"

        st.success(f"Prediction: {predicted_label.upper()}")
