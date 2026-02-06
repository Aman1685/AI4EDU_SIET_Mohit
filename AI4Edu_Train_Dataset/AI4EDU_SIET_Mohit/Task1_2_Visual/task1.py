import os
import cv2
import torch
import torch.nn as nn
import pandas as pd
import mediapipe as mp

from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIDEO_DIR = "/nlsasfs/home/gpucbh/vyakti13/AI4Edu_Train_Dataset/Train_fixed"
LABEL_FILE = "/nlsasfs/home/gpucbh/vyakti13/AI4Edu_Train_Dataset/labels_train.xlsx"
VISUAL_FEATURE_DIR = "./visual_features"

os.makedirs(VISUAL_FEATURE_DIR, exist_ok=True)

# =========================
# LABEL MAPPING
# =========================
def map_label(label):
    return 0 if label <= 0.33 else 1

# =========================
# FACE DETECTION
# =========================
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.3  # lower to detect more faces
)

def detect_face(frame):
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
    return face

# =========================
# RESNET FEATURE EXTRACTOR
# =========================
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet = resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_frame_feature(frame):
    face = detect_face(frame)
    if face is None:
        face = frame  # fallback to full frame
    face = transform(face).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(face)
    return feat.squeeze(0)

def extract_video_features(video_path, frame_skip=5, max_frames=60):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return None

    features = []
    frame_id = 0
    while cap.isOpened() and len(features) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None or frame.size == 0:
            frame_id += 1
            continue
        # Always extract, no skipping for debug
        feat = extract_frame_feature(frame)
        if feat is not None:
            features.append(feat)
        frame_id += 1

    cap.release()
    if len(features) == 0:
        print(f"[WARN] No features extracted for {video_path}")
        return None

    print(f"[INFO] Extracted {len(features)} features for {video_path}")
    return torch.stack(features)

def find_video_file(video_name):
    base = os.path.splitext(str(video_name).strip())[0]
    for file in os.listdir(VIDEO_DIR):
        if os.path.splitext(file)[0] == base:
            return os.path.join(VIDEO_DIR, file)
    return None

# =========================
# STEP 1: EXTRACT VISUAL FEATURES
# =========================
print("Step 1: Extracting visual features...")
df = pd.read_excel(LABEL_FILE)

for idx, row in df.iterrows():
    video_name = str(row[0]).strip()
    video_path = find_video_file(video_name)
    if video_path is None:
        print(f"[WARN] Video not found: {video_name}")
        continue

    features = extract_video_features(video_path)
    if features is None:
        print(f"[WARN] Skipping {video_name}, no features")
        continue

    save_path = os.path.join(VISUAL_FEATURE_DIR, f"{video_name}.pt")
    torch.save(features, save_path)
    print(f"[SAVED] Visual features for {video_name} ({features.shape})")

print("✅ Visual feature extraction complete!\n")

# =========================
# STEP 2: DATASET
# =========================
class AttentivenessDataset(Dataset):
    def __init__(self, label_file, feature_dir):
        self.df = pd.read_excel(label_file)
        self.feature_dir = feature_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_name = str(row.iloc[0]).strip()
        label_val = row.iloc[1]
        label = map_label(label_val)

        feat_path = os.path.join(self.feature_dir, f"{video_name}.pt")
        if not os.path.exists(feat_path):
            print(f"[WARN] Feature file missing: {feat_path}")
            return None

        features = torch.load(feat_path)
        return features, torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    features, labels = zip(*batch)
    features = pad_sequence(features, batch_first=True)
    labels = torch.stack(labels)
    return features, labels

dataset = AttentivenessDataset(LABEL_FILE, VISUAL_FEATURE_DIR)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0)

# =========================
# STEP 3: MODEL
# =========================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim*2, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        return (weights * lstm_out).sum(dim=1)

class AttentivenessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )
        self.attention = TemporalAttention(256)
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.attention(lstm_out)
        return self.fc(x)

# =========================
# STEP 4: TRAINING
# =========================
model = AttentivenessModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()

EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_acc = 0
    steps = 0
    for batch in loader:
        if batch is None:
            continue
        features, labels = batch
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += compute_accuracy(outputs, labels)
        steps += 1

    if steps > 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/steps:.4f} | Acc: {total_acc/steps*100:.2f}%")
    else:
        print("[WARN] No valid batches!")

# =========================
# STEP 5: SAMPLE PREDICTION
# =========================
model.eval()
with torch.no_grad():
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample is None:
            continue
        sample_feat, _ = sample
        sample_feat = sample_feat.unsqueeze(0).to(device)
        pred = torch.argmax(model(sample_feat), dim=1).item()
        print(f"{df.iloc[idx,0]} -> {'High Attentiveness' if pred==1 else 'Low Attentiveness'}")
        break  # first sample only
