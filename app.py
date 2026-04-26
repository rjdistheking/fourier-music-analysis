import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import plotly.graph_objs as go
from scipy.fft import rfft, irfft, rfftfreq

st.set_page_config(layout="wide", page_title="Music Fourier Analyzer")

st.title("🎵 Interactive Audio Spectrum Analyzer")
st.markdown("Yeh app audio ko decompose aur reconstruct karta hai!")

uploaded_file = st.file_uploader("Upload an Audio File", type=['wav', 'mp3', 'aac'])

if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, sr=None, mono=True)
    st.audio(uploaded_file, format='audio/wav')
    
    # 1. Waveform Plot
    time = np.linspace(0, len(y) / sr, num=len(y))
    fig1 = go.Figure(data=go.Scatter(x=time[::100], y=y[::100], mode='lines'))
    fig1.update_layout(title="Original Time Domain Signal", xaxis_title="Time (s)", yaxis_title="Amplitude")
    st.plotly_chart(fig1, use_container_width=True)

    # 2. FFT Decomposition
    N = len(y)
    yf = rfft(y) 
    xf = rfftfreq(N, 1 / sr)
    magnitude = np.abs(yf)
    
    fig2 = go.Figure(data=go.Scatter(x=xf[::50], y=magnitude[::50], mode='lines', marker=dict(color='orange')))
    fig2.update_layout(title="Frequency Spectrum", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Filter Slider
    max_freq = int(xf[-1])
    filter_range = st.slider("Select Frequency Range to KEEP:", 0, max_freq, (0, max_freq))
    
    yf_filtered = yf.copy()
    yf_filtered[(xf < filter_range[0]) | (xf > filter_range[1])] = 0
    
    # 4. Reconstruction
    if st.button("Reconstruct Audio"):
        y_reconstructed = irfft(yf_filtered)
        if len(y_reconstructed) > len(y): y_reconstructed = y_reconstructed[:len(y)]
        
        error = np.mean((y[:len(y_reconstructed)] - y_reconstructed) ** 2)
        st.metric(label="Reconstruction MSE", value=f"{error:.6f}")

        sf.write("reconstructed.wav", y_reconstructed, sr)
        st.audio("reconstructed.wav", format='audio/wav')
