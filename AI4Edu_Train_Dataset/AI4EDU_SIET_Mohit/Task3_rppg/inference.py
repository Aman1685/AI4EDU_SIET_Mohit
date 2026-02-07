import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------- CONFIG --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_VIDEO_DIR = "./Test_fixed"           # Folder with test videos (.mp4)
OUTPUT_DIR = "./rppg_test_outputs"        # Where rPPG CSVs will be saved
VISUAL_FOLDER = "./Task1_2_Visual/visual_features"  # Folder with visual features (.pt)
MODEL_PATH = "./Task3_rppg/model.pth"    # DeepPhys pretrained model
LABELS_PATH = "./labels_test.xlsx"       # Optional: test labels in 'label' column

MLP_VIS_PATH = "./mlp_vis.pkl"
MLP_RPPG_PATH = "./mlp_rppg.pkl"
MLP_EF_PATH = "./mlp_ef.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- MODEL DEFINITIONS --------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c):
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
    def forward(self,app, mot):
        a = self.appearance(app)
        m = self.motion(mot)
        x = torch.cat([a,m], dim=1)
        return self.fc(x)

# Load DeepPhys model
deepphys = DeepPhys().to(DEVICE)
deepphys.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
deepphys.eval()

# -------------------- VIDEO PROCESSING --------------------
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128,128))
        frame_tensor = torch.tensor(frame/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0)
        frames.append(frame_tensor)
    cap.release()
    frames = torch.cat(frames, dim=0).to(DEVICE)
    
    # Motion input: frame differences
    motion = torch.zeros_like(frames)
    motion[1:] = frames[1:] - frames[:-1]

    # DeepPhys inference
    with torch.no_grad():
        rppg_signal = deepphys(frames, motion).squeeze().cpu().numpy()
    
    # Save rPPG CSV
    out_file = os.path.join(OUTPUT_DIR, os.path.basename(video_path).replace('.mp4','.csv'))
    pd.DataFrame({'rppg':rppg_signal}).to_csv(out_file, index=False)

# Process all test videos
print("Processing test videos for rPPG extraction...")
for v in sorted(os.listdir(TEST_VIDEO_DIR)):
    if v.endswith(".mp4"):
        video_path = os.path.join(TEST_VIDEO_DIR, v)
        print("Processing:", v)
        process_video(video_path)
print("rPPG extraction completed.")

# -------------------- LOAD rPPG FEATURES --------------------
rppg_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')])
rppg_features_list = []
for f in rppg_files:
    df = pd.read_csv(os.path.join(OUTPUT_DIR, f))
    sig = df['rppg'].values
    feats = [sig.mean(), sig.std(), sig.max()-sig.min()]  # simple stats
    rppg_features_list.append(feats)
rppg_features = np.array(rppg_features_list)

# -------------------- LOAD VISUAL FEATURES --------------------
visual_files = sorted([f for f in os.listdir(VISUAL_FOLDER) if f.endswith('.pt')])
visual_features_list = []
for f in visual_files:
    tensor = torch.load(os.path.join(VISUAL_FOLDER,f))
    if isinstance(tensor, dict) and 'features' in tensor:
        tensor = tensor['features']
    arr = tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else np.array(tensor)
    if len(arr.shape) > 1:
        arr = arr.mean(axis=0)  # average if multi-dimensional
    visual_features_list.append(arr)
visual_features = np.vstack(visual_features_list)

# -------------------- LOAD LABELS (OPTIONAL) --------------------
if os.path.exists(LABELS_PATH):
    labels_df = pd.read_excel(LABELS_PATH)
    labels = labels_df['label'].values.flatten()
else:
    labels = None

# -------------------- ALIGN DATA --------------------
min_len = min(rppg_features.shape[0], visual_features.shape[0], len(labels) if labels is not None else np.inf)
rppg_features = rppg_features[:min_len]
visual_features = visual_features[:min_len]
if labels is not None:
    labels = labels[:min_len]

# -------------------- FEATURE SCALING --------------------
scaler_rppg = StandardScaler()
scaler_vis = StandardScaler()
rppg_scaled = scaler_rppg.fit_transform(rppg_features)
visual_scaled = scaler_vis.fit_transform(visual_features)
early_fusion = np.hstack([visual_scaled, rppg_scaled])

# -------------------- LOAD MLP MODELS --------------------
model_vis = joblib.load(MLP_VIS_PATH)
model_rppg = joblib.load(MLP_RPPG_PATH)
model_ef = joblib.load(MLP_EF_PATH)

# -------------------- PREDICTIONS --------------------
y_pred_vis = model_vis.predict(visual_scaled)
y_pred_rppg = model_rppg.predict(rppg_scaled)
y_pred_ef = model_ef.predict(early_fusion)

# -------------------- RESULTS --------------------
if labels is not None:
    from sklearn.metrics import accuracy_score
    print("Visual Only Accuracy:", accuracy_score(labels, y_pred_vis)*100)
    print("rPPG Only Accuracy:", accuracy_score(labels, y_pred_rppg)*100)
    print("Early Fusion Accuracy:", accuracy_score(labels, y_pred_ef)*100)
else:
    print("Predictions done. Labels not provided.")
    # Optionally save predictions
    pd.DataFrame({
        "video": [os.path.basename(f).replace('.csv','.mp4') for f in rppg_files],
        "pred_vis": y_pred_vis,
        "pred_rppg": y_pred_rppg,
        "pred_ef": y_pred_ef
    }).to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
    print("Predictions saved to predictions.csv")
