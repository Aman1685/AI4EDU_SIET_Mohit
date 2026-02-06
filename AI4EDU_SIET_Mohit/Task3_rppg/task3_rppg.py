import os
import cv2
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mediapipe as mp
from scipy.signal import butter, filtfilt


# =========================
# CONFIG
# =========================
VIDEO_DIR = "/nlsasfs/home/gpucbh/vyakti13/AI4Edu_Train_Dataset/Train_fixed"
OUTPUT_DIR = "./rppg_outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_FRAMES = 900
WINDOW_SIZE = 32   # DeepPhys temporal window
STRIDE = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# FACE DETECTION
# =========================
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.4)

def detect_face(frame, size=(128, 128)):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    if not results.detections:
        return None

    bbox = results.detections[0].location_data.relative_bounding_box
    h, w, _ = frame.shape

    x1 = max(0, int(bbox.xmin * w))
    y1 = max(0, int(bbox.ymin * h))
    x2 = min(w, int((bbox.xmin + bbox.width) * w))
    y2 = min(h, int((bbox.ymin + bbox.height) * h))

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None

    return cv2.resize(face, size)

# =========================
# BANDPASS FILTER
# =========================
def bandpass(signal, fs, low=0.7, high=4.0):
    b, a = butter(3, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, signal)

# =========================
# CLASSICAL rPPG
# =========================
def green_rppg(rgb, fs):
    return bandpass(rgb[:,1], fs)

def chrom_rppg(rgb, fs):
    R, G, B = rgb[:,0], rgb[:,1], rgb[:,2]
    X = 3*R - 2*G
    Y = 1.5*R + G - 1.5*B
    alpha = np.std(X)/(np.std(Y)+1e-6)
    return bandpass(X - alpha*Y, fs)

def pos_rppg(rgb, fs):
    rgb = rgb / np.mean(rgb, axis=0)
    X = 3*rgb[:,0] - 2*rgb[:,1]
    Y = 1.5*rgb[:,0] + rgb[:,1] - 1.5*rgb[:,2]
    alpha = np.std(X)/(np.std(Y)+1e-6)
    return bandpass(X - alpha*Y, fs)

# =========================
# DEEPPHYS MODEL
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

    def forward(self, x):
        return self.net(x).squeeze(-1).squeeze(-1)

class DeepPhys(nn.Module):
    def __init__(self):
        super().__init__()
        self.appearance = ConvBlock(3)
        self.motion = ConvBlock(3)
        self.fc = nn.Linear(64, 1)

    def forward(self, app, mot):
        a = self.appearance(app)
        m = self.motion(mot)
        x = torch.cat([a, m], dim=1)
        return self.fc(x)

deepphys = DeepPhys().to(DEVICE)
deepphys.eval()  # inference-only for Task 3

# =========================
# DEEPPHYS SIGNAL EXTRACTION
# =========================
def extract_deepphys_signal(frames):
    app = []
    mot = []

    for i in range(1, len(frames)):
        app.append(frames[i] / 255.0)
        mot.append((frames[i] - frames[i-1]) / 255.0)

    app = np.array(app)
    mot = np.array(mot)

    signals = []

    for i in range(0, len(app) - WINDOW_SIZE, STRIDE):
        a = torch.from_numpy(app[i:i+WINDOW_SIZE]).permute(0,3,1,2).float().to(DEVICE)
        m = torch.from_numpy(mot[i:i+WINDOW_SIZE]).permute(0,3,1,2).float().to(DEVICE)

        with torch.no_grad():
            rppg = deepphys(a, m).mean().item()
            signals.append(rppg)

    return np.array(signals)

# =========================
# MAIN VIDEO PROCESSOR
# =========================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    frames = []
    rgb_signal = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        face = detect_face(frame)
        if face is None:
            continue

        frames.append(face)
        rgb_signal.append(face.mean(axis=(0,1)))

        if len(frames) >= MAX_FRAMES:
            break

    cap.release()

    if len(frames) < 100:
        print("Skipped (too few face frames)")
        return

    rgb_signal = np.array(rgb_signal)
    name = os.path.splitext(os.path.basename(video_path))[0]

    signals = {
        "GREEN": green_rppg(rgb_signal, fps),
        "CHROM": chrom_rppg(rgb_signal, fps),
        "POS": pos_rppg(rgb_signal, fps),
        "DEEPHYS": extract_deepphys_signal(frames)
    }

    for k, sig in signals.items():
        if sig is None or len(sig) == 0:
            continue

        pd.DataFrame({"rppg": sig}).to_csv(
            f"{OUTPUT_DIR}/{name}_{k}.csv", index=False
        )

        with open(f"{OUTPUT_DIR}/{name}_{k}.json", "w") as f:
            json.dump(sig.tolist(), f)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    for v in os.listdir(VIDEO_DIR):
        if v.endswith(".mp4"):
            print("Processing:", v)
            process_video(os.path.join(VIDEO_DIR, v))

    print("\n✅ Classical + DeepPhys rPPG extraction complete")