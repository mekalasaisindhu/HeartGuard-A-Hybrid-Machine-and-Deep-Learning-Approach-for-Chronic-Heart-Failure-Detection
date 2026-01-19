import streamlit as st
import requests
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import io

# BACKEND URL (IMPORTANT FIX)
BACKEND_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(page_title="HeartGuard – CHF Detection", layout="wide")

st.title("HeartGuard: Chronic Heart Failure Detection")
st.write("Upload a PCG (heart sound) recording to analyze heart health.")

# ===================================================================
# FILE UPLOAD
# ===================================================================
uploaded_file = st.file_uploader("Upload PCG File (.wav / .mp3)", type=["wav", "mp3"])

if uploaded_file:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format="audio/wav")

    audio_buffer = io.BytesIO(audio_bytes)

    # Load signal for visualization
    signal, sr = librosa.load(audio_buffer, sr=2000)

    # Waveform and filtered plots
    st.subheader("Signal Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Waveform")
        fig, ax = plt.subplots(figsize=(6, 3))
        librosa.display.waveshow(signal, sr=sr, ax=ax)
        st.pyplot(fig)

    # Apply simple filter
    from scipy.signal import butter, filtfilt

    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    b, a = butter_bandpass(30, 600, sr)
    filtered = filtfilt(b, a, signal)

    with col2:
        st.write("Filtered Signal")
        fig, ax = plt.subplots(figsize=(6, 3))
        librosa.display.waveshow(filtered, sr=sr, ax=ax)
        st.pyplot(fig)

    # MEL SPECTROGRAM
    st.subheader("Mel Spectrogram")

    mel = librosa.feature.melspectrogram(y=filtered, sr=sr, n_mels=128, fmax=800)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    st.pyplot(fig)

    # Reset buffer for sending to backend
    uploaded_file.seek(0)

    # ===================================================================
    # ANALYZE BUTTON
    # ===================================================================
    if st.button("Analyze"):
        st.subheader("Prediction Results")

        with st.spinner("Analyzing PCG signal..."):
            files = {"file": (uploaded_file.name, audio_bytes, uploaded_file.type)}
            try:
                response = requests.post(BACKEND_URL, files=files, timeout=20)
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")
                st.stop()

        if response.status_code != 200:
            st.error("Backend returned an error.")
            st.write(response.text)
            st.stop()

        # Parse backend result
        result = response.json()

        # Prediction
        label = result["prediction"]
        confidence = result["confidence"]

        pred_col, feat_col = st.columns(2)

        with pred_col:
            st.write("Prediction")
            if label == 1:
                st.markdown("### Detected: **Chronic Heart Failure**")
            else:
                st.markdown("### Detected: **Normal Heart Sound**")
            st.write(f"Confidence: {confidence * 100:.2f}%")

        with feat_col:
            st.write("Top Contributing Features")
            for feat_name, feat_value in result["top_features"].items():
                st.write(f"• **{feat_name}:** {feat_value}")

        st.success("Analysis complete!")