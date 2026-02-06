import os
import cv2
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mediapipe as mp
from scipy.signal import butter, filtfilt, welch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
VIDEO_DIR = "/nlsasfs/home/gpucbh/vyakti13/AI4Edu_Train_Dataset/Train_fixed"
VISUAL_FEATURES_FILE = "./visual_features.npy"  # Task 1 features (num_videos x feature_dim)
LABELS_FILE = "./labels.npy"  # e.g., cognitive state: 0/1
OUTPUT_DIR = "./rppg_outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_FRAMES = 900
WINDOW_SIZE = 32
STRIDE = 1
FPS = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# FACE DETECTION
# =========================
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.4)

def detect_face(frame, size=(128,128)):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    if not results.detections:
        return None
    bbox = results.detections[0].location_data.relative_bounding_box
    h, w, _ = frame.shape
    x1 = max(0,int(bbox.xmin*w))
    y1 = max(0,int(bbox.ymin*h))
    x2 = min(w,int((bbox.xmin+bbox.width)*w))
    y2 = min(h,int((bbox.ymin+bbox.height)*h))
    face = frame[y1:y2, x1:x2]
    if face.size==0: return None
    return cv2.resize(face,size)

# =========================
# BANDPASS FILTER
# =========================
def bandpass(signal, fs, low=0.7, high=4.0):
    b,a = butter(3,[low/(fs/2), high/(fs/2)],btype="band")
    return filtfilt(b,a,signal)

# =========================
# CLASSICAL rPPG METHODS
# =========================
def green_rppg(rgb, fs):
    return bandpass(rgb[:,1], fs)

def chrom_rppg(rgb, fs):
    R,G,B = rgb[:,0], rgb[:,1], rgb[:,2]
    X = 3*R-2*G
    Y = 1.5*R + G -1.5*B
    alpha = np.std(X)/(np.std(Y)+1e-6)
    return bandpass(X - alpha*Y, fs)

def pos_rppg(rgb, fs):
    rgb = rgb / np.mean(rgb, axis=0)
    X = 3*rgb[:,0]-2*rgb[:,1]
    Y = 1.5*rgb[:,0]+rgb[:,1]-1.5*rgb[:,2]
    alpha = np.std(X)/(np.std(Y)+1e-6)
    return bandpass(X-alpha*Y, fs)

# =========================
# DEEPPHYS MODEL
# =========================
class ConvBlock(nn.Module):
    def __init__(self,in_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c,32,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
    def forward(self,x):
        return self.net(x).squeeze(-1).squeeze(-1)

class DeepPhys(nn.Module):
    def __init__(self):
        super().__init__()
        self.appearance = ConvBlock(3)
        self.motion = ConvBlock(3)
        self.fc = nn.Linear(64,1)
    def forward(self, app,mot):
        a = self.appearance(app)
        m = self.motion(mot)
        x = torch.cat([a,m],dim=1)
        return self.fc(x)

deepphys = DeepPhys().to(DEVICE)
deepphys.eval()

# =========================
# DEEPPHYS SIGNAL EXTRACTION
# =========================
def extract_deepphys_signal(frames):
    app, mot = [], []
    for i in range(1,len(frames)):
        app.append(frames[i]/255.0)
        mot.append((frames[i]-frames[i-1])/255.0)
    app = np.array(app); mot = np.array(mot)
    signals = []
    for i in range(0,len(app)-WINDOW_SIZE,STRIDE):
        a = torch.from_numpy(app[i:i+WINDOW_SIZE]).permute(0,3,1,2).float().to(DEVICE)
        m = torch.from_numpy(mot[i:i+WINDOW_SIZE]).permute(0,3,1,2).float().to(DEVICE)
        with torch.no_grad():
            rppg = deepphys(a,m).mean().item()
            signals.append(rppg)
    return np.array(signals)

# =========================
# PHYSIOLOGICAL FEATURE EXTRACTION
# =========================
def extract_physio_features(rppg, fs):
    signal = rppg - np.mean(rppg)
    n = len(signal)
    if n < 2: return np.zeros(4)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_signal = np.abs(np.fft.rfft(signal))
    idx = np.where((freqs>=0.7)&(freqs<=4.0))
    freqs = freqs[idx]; fft_signal=fft_signal[idx]
    if len(fft_signal)==0: return np.zeros(4)
    peak_freq = freqs[np.argmax(fft_signal)]
    hr = peak_freq*60
    rr_intervals = 60/(peak_freq + 1e-6)
    hrv = np.std(rr_intervals) if np.size(rr_intervals)>1 else 0.0
    sqi = np.max(fft_signal)/(np.sum(fft_signal)+1e-6)
    f,Pxx = welch(signal, fs=fs, nperseg=min(256,len(signal)))
    lf = np.sum(Pxx[(f>=0.04)&(f<0.15)])
    hf = np.sum(Pxx[(f>=0.15)&(f<0.4)])
    lfhf = lf/(hf+1e-6)
    return np.array([hr, hrv, sqi, lfhf])

# =========================
# VIDEO PROCESSOR + FEATURE FUSION
# =========================
def process_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    rgb_signal = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        face = detect_face(frame)
        if face is None: continue
        frames.append(face)
        rgb_signal.append(face.mean(axis=(0,1)))
        if len(frames)>=MAX_FRAMES: break
    cap.release()
    if len(frames)<100: return None,None
    rgb_signal = np.array(rgb_signal)

    rppg_dict = {
        "GREEN": green_rppg(rgb_signal,FPS),
        "CHROM": chrom_rppg(rgb_signal,FPS),
        "POS": pos_rppg(rgb_signal,FPS),
        "DEEPHYS": extract_deepphys_signal(frames)
    }

    physio_features = []
    for sig in rppg_dict.values():
        if sig is None or len(sig)==0: continue
        physio_features.append(extract_physio_features(sig,FPS))
    if len(physio_features)==0: return None,None
    physio_features = np.mean(np.array(physio_features),axis=0)

    return physio_features, rgb_signal

# =========================
# MAIN LOOP
# =========================
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
visual_features = np.load(VISUAL_FEATURES_FILE)
labels = np.load(LABELS_FILE)

physio_features_list = []
visual_list = []

for idx, v in enumerate(video_files):
    print(f"Processing video {idx+1}/{len(video_files)}: {v}")
    try:
        physio_feat, visual_feat = process_video_features(os.path.join(VIDEO_DIR,v))
        if physio_feat is None:
            print("  Skipped: too few frames or face not detected")
            continue
        physio_features_list.append(physio_feat)
        visual_list.append(visual_features[idx])
        print("  Success")
    except Exception as e:
        print(f"  Error: {e}")

physio_features_array = np.array(physio_features_list)
visual_array = np.array(visual_list)
labels_array = labels[:len(physio_features_array)]

# ---- Early Fusion ----
X_early = np.concatenate([visual_array, physio_features_array], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_early, labels_array, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred_early = clf.predict(X_test)
acc_early = accuracy_score(y_test, y_pred_early)

# ---- Visual Only ----
X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(visual_array, labels_array, test_size=0.2, random_state=42)
clf_v = RandomForestClassifier(n_estimators=200, random_state=42)
clf_v.fit(X_train_v, y_train_v)
y_pred_v = clf_v.predict(X_test_v)
acc_visual = accuracy_score(y_test_v, y_pred_v)

# ---- Physiological Only ----
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(physio_features_array, labels_array, test_size=0.2, random_state=42)
clf_p = RandomForestClassifier(n_estimators=200, random_state=42)
clf_p.fit(X_train_p, y_train_p)
y_pred_p = clf_p.predict(X_test_p)
acc_physio = accuracy_score(y_test_p, y_pred_p)

print(f"Visual Only Accuracy: {acc_visual*100:.2f}%")
print(f"Physio Only Accuracy: {acc_physio*100:.2f}%")
print(f"Early Fusion Accuracy: {acc_early*100:.2f}%")

# =========================
# PERFORMANCE BAR PLOT
# =========================
plt.figure(figsize=(6,4))
methods = ["Visual Only","Physio Only","Early Fusion"]
accs = [acc_visual*100, acc_physio*100, acc_early*100]
plt.bar(methods, accs, color=["blue","green","orange"])
plt.ylabel("Accuracy (%)")
plt.title("Ablation Study: Visual vs Physio vs Fusion")
plt.ylim(0,100)
for i,v in enumerate(accs):
    plt.text(i,v+1,f"{v:.2f}%", ha="center")
plt.show()
