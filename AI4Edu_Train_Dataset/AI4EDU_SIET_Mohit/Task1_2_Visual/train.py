import os
import cv2
import torch
import torch.nn as nn
import pandas as pd
import mediapipe as mp
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIDEO_DIR = "/nlsasfs/home/gpucbh/vyakti13/AI4Edu_Train_Dataset/Train_fixed"
LABEL_FILE = "/nlsasfs/home/gpucbh/vyakti13/AI4Edu_Train_Dataset/labels_train.xlsx"

NUM_CLASSES = 4
CLASS_NAMES = {0: "Disengaged", 1: "Distracted", 2: "Nominally Engaged", 3: "Highly Engaged"}

# =========================
# LABEL MAPPING
# =========================
def map_label_multiclass(label):
    if label <= 0.25:
        return 0
    elif label <= 0.5:
        return 1
    elif label <= 0.75:
        return 2
    else:
        return 3

# =========================
# FACE DETECTOR
# =========================
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.4)

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
    return frame[y1:y2, x1:x2]

# =========================
# FEATURE EXTRACTOR
# =========================
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet = resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.455, 0.456, 0.406], std=[0.220, 0.224, 0.225])
])

def extract_frame_feature(frame):
    face = detect_face(frame)
    if face is None:
        face = frame
    face = transform(face).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(face)
    return feat.squeeze(0)

def extract_video_features(video_path, frame_skip=5, max_frames=60):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    features = []
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame.mean() < 3:
            frame_id += 1
            continue
        if frame_id % frame_skip == 0:
            features.append(extract_frame_feature(frame))
        if len(features) >= max_frames:
            break
        frame_id += 1
    cap.release()
    if len(features) == 0:
        return None
    return torch.stack(features)

# =========================
# VIDEO INDEX
# =========================
def build_video_index(video_dir):
    index = {}
    for f in os.listdir(video_dir):
        key = "_".join(os.path.splitext(f.lower())[0].split("_")[:3])
        index.setdefault(key, []).append(os.path.join(video_dir, f))
    return index

# =========================
# DATASET
# =========================
class AttentivenessDataset(Dataset):
    def __init__(self, video_dir, label_file):
        self.df = pd.read_excel(label_file)
        self.video_index = build_video_index(video_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        key = "_".join(os.path.splitext(str(row.iloc[0]).lower())[0].split("_")[:3])
        videos = self.video_index.get(key)
        if not videos:
            return None
        features = extract_video_features(videos[0])
        if features is None:
            return None
        label = map_label_multiclass(row.iloc[1])
        return features, torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    features, labels = zip(*batch)
    return pad_sequence(features, batch_first=True), torch.stack(labels)

# =========================
# MODEL
# =========================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (w * x).sum(dim=1)

class AttentivenessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, batch_first=True, bidirectional=True)
        self.attn = TemporalAttention(256)
        self.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.attn(x)
        return self.fc(x)

# =========================
# TRAIN/VALIDATION SPLIT
# =========================
dataset = AttentivenessDataset(VIDEO_DIR, LABEL_FILE)

# Split 80% train, 20% validation
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# =========================
# TRAINING
# =========================
model = AttentivenessModel().to(device)
class_weights = torch.tensor([1.5, 1.2, 1.0, 1.5]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def accuracy(logits, labels):
    return (torch.argmax(logits, 1) == labels).float().mean().item()

EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    loss_sum, acc_sum, steps = 0, 0, 0
    for batch in train_loader:
        if batch is None:
            continue
        x, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        acc_sum += accuracy(out, y)
        steps += 1
    train_loss = loss_sum / steps if steps > 0 else 0
    train_acc = acc_sum / steps if steps > 0 else 0

    # Validation
    model.eval()
    val_loss_sum, val_acc_sum, val_steps = 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            x_val, y_val = batch
            x_val, y_val = x_val.to(device), y_val.to(device)
            out_val = model(x_val)
            loss_val = criterion(out_val, y_val)
            val_loss_sum += loss_val.item()
            val_acc_sum += accuracy(out_val, y_val)
            val_steps += 1
    val_loss = val_loss_sum / val_steps if val_steps > 0 else 0
    val_acc = val_acc_sum / val_steps if val_steps > 0 else 0

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

# =========================
# FINAL PREDICTION (SAMPLE)
# =========================
model.eval()
with torch.no_grad():
    sample = dataset[0]
    if sample is not None:
        feat, _ = sample
        pred = torch.argmax(model(feat.unsqueeze(0).to(device)), 1).item()
        print("\nFinal Sample Prediction:", CLASS_NAMES[pred])
