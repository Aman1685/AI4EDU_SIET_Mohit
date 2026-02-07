# inference.py

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from model import load_model

# -----------------------------
# 1️⃣ rPPG feature extraction
# -----------------------------
from scipy.signal import find_peaks, welch

def extract_rppg_features(rppg_signal, fs=30):
    features = {}
    peaks, _ = find_peaks(rppg_signal, distance=fs*0.5)
    rr_intervals = np.diff(peaks) / fs

    if len(rr_intervals) < 2 or np.all(rr_intervals == 0):
        features = {k:0 for k in ['HR','SDNN','RMSSD','LF','HF','LF_HF','SQI']}
        return features

    features['HR'] = 60 / np.mean(rr_intervals)
    features['SDNN'] = np.std(rr_intervals)
    features['RMSSD'] = np.sqrt(np.mean(np.diff(rr_intervals)**2))

    nperseg = min(256, len(rr_intervals))
    f, Pxx = welch(rr_intervals, fs=fs, nperseg=nperseg)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)
    LF = np.trapz(Pxx[(f>=lf_band[0]) & (f<=lf_band[1])], f[(f>=lf_band[0]) & (f<=lf_band[1])])
    HF = np.trapz(Pxx[(f>=hf_band[0]) & (f<=hf_band[1])], f[(f>=hf_band[0]) & (f<=hf_band[1])])
    features['LF'] = LF
    features['HF'] = HF
    features['LF_HF'] = LF/HF if HF != 0 else 0

    expected_beats = len(rppg_signal)/fs * (features['HR']/60)
    features['SQI'] = min(len(peaks)/expected_beats, 1.0)
    return np.array(list(features.values()))

# -----------------------------
# 2️⃣ Load CSV features
# -----------------------------
def load_visual_csv(file_path):
    df = pd.read_csv(file_path)
    arr = df.values.flatten()
    if len(arr.shape) > 1:
        arr = arr.mean(axis=0)
    return arr

def load_rppg_csv(file_path):
    df = pd.read_csv(file_path)
    if 'rppg' in df.columns:
        signal = df['rppg'].values
    else:
        signal = df.values.flatten()
    return extract_rppg_features(signal)

# -----------------------------
# 3️⃣ Prediction function
# -----------------------------
def predict(model_path, visual_csv=None, rppg_csv=None, scaler_vis=None, scaler_rppg=None, num_classes=3, device='cpu'):
    """
    Predict class from visual and/or rPPG CSV files.
    """
    features = []

    # Visual features
    if visual_csv is not None:
        vis_feat = load_visual_csv(visual_csv)
        if scaler_vis is not None:
            vis_feat = scaler_vis.transform([vis_feat])[0]
        features.append(vis_feat)

    # rPPG features
    if rppg_csv is not None:
        rppg_feat = load_rppg_csv(rppg_csv)
        if scaler_rppg is not None:
            rppg_feat = scaler_rppg.transform([rppg_feat])[0]
        features.append(rppg_feat)

    if len(features) == 0:
        raise ValueError("Provide at least visual_csv or rppg_csv.")

    # Combine features (early fusion)
    input_feat = np.hstack(features)
    input_tensor = torch.tensor(input_feat, dtype=torch.float32).unsqueeze(0)  # batch of 1

    model = load_model(model_path, input_dim=input_feat.shape[0], num_classes=num_classes, device=device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]
        pred_class = np.argmax(probs)

    return pred_class, probs

# -----------------------------
# 4️⃣ Example usage
# -----------------------------
if __name__ == "__main__":
    # Load pre-trained scalers (or fit on training data)
    # For demonstration, we fit on dummy data
    scaler_vis = StandardScaler()
    scaler_rppg = StandardScaler()

    # Dummy fitting (replace with actual training scaler saved via pickle if possible)
    scaler_vis.fit(np.random.rand(10, 128))  # 128 visual features example
    scaler_rppg.fit(np.random.rand(10, 7))   # 7 rPPG features

    visual_csv_path = 'example_visual.csv'  # replace with actual
    rppg_csv_path = 'example_rppg.csv'      # replace with actual
    model_path = 'model_ef.pth'             # replace with your trained model

    pred_class, probs = predict(model_path, visual_csv=visual_csv_path, rppg_csv=rppg_csv_path,
                                scaler_vis=scaler_vis, scaler_rppg=scaler_rppg, num_classes=3)
    print("Predicted class:", pred_class)
    print("Probabilities:", probs)
