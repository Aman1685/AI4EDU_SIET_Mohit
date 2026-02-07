import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Robust rPPG feature extraction
# -----------------------------
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
    return features

# -----------------------------
# 2️⃣ Load rPPG CSVs
# -----------------------------
rppg_folder = '/nlsasfs/home/gpucbh/vyakti13/AI4Edu_Train_Dataset/AI4EDU_SIET_Mohit/Task3_rppg/rppg_outputs'
rppg_files = sorted([f for f in os.listdir(rppg_folder) if f.endswith('.csv')])
rppg_features_list = []

for f in rppg_files:
    file_path = os.path.join(rppg_folder, f)
    try:
        df = pd.read_csv(file_path)
        if df.shape[0] == 0:
            print(f"[WARNING] Empty CSV skipped: {f}")
            continue
        signal = df['rppg'].values if 'rppg' in df.columns else df.values.flatten()
        feats = extract_rppg_features(signal)
        rppg_features_list.append(list(feats.values()))
    except pd.errors.EmptyDataError:
        print(f"[WARNING] Empty CSV skipped: {f}")
        continue
    except Exception as e:
        print(f"[ERROR] Failed to read {f}: {e}")
        continue

if len(rppg_features_list) == 0:
    raise ValueError("No rPPG features found! Check your CSV files.")

rppg_features = np.array(rppg_features_list)
print("rPPG features shape:", rppg_features.shape)

# -----------------------------
# 3️⃣ Load visual features from CSV instead of .pt
# -----------------------------
visual_csv_folder = '/nlsasfs/home/gpucbh/vyakti13/AI4Edu_Train_Dataset/AI4EDU_SIET_Mohit/Task1_2_Visual/visual_features_csv'  # folder containing visual CSVs
visual_features_list = []

visual_csv_files = sorted([f for f in os.listdir(visual_csv_folder) if f.endswith('.csv')])

for f in visual_csv_files:
    file_path = os.path.join(visual_csv_folder, f)
    try:
        df = pd.read_csv(file_path)
        if df.shape[0] == 0:
            print(f"[WARNING] Empty CSV skipped: {f}")
            continue
        arr = df.values.flatten()
        # If multi-dimensional, take mean across rows
        if len(arr.shape) > 1:
            arr = arr.mean(axis=0)
        visual_features_list.append(arr)
    except pd.errors.EmptyDataError:
        print(f"[WARNING] Empty CSV skipped: {f}")
        continue
    except Exception as e:
        print(f"[ERROR] Failed to read {f}: {e}")

if len(visual_features_list) == 0:
    raise ValueError("No visual features found! Check your CSV files.")

visual_features = np.vstack(visual_features_list)
print("Visual features shape:", visual_features.shape)

# -----------------------------
# 4️⃣ Load labels
# -----------------------------
labels_df = pd.read_excel('/nlsasfs/home/gpucbh/vyakti13/AI4Edu_Train_Dataset/labels_train.xlsx')  # update path as needed
labels = np.array(labels_df['label'].values).flatten()
print("Original labels shape:", labels.shape)

# -----------------------------
# 5️⃣ Align arrays
# -----------------------------
min_len = min(rppg_features.shape[0], visual_features.shape[0], labels.shape[0])
rppg_features = rppg_features[:min_len]
visual_features = visual_features[:min_len]
labels = labels[:min_len]

# -----------------------------
# 6️⃣ Map labels to integers
# -----------------------------
unique_vals = np.unique(labels)
label_map = {val: idx for idx, val in enumerate(unique_vals)}
labels_int = np.array([label_map[v] for v in labels])
num_classes = len(unique_vals)
print("Number of classes:", num_classes)

# -----------------------------
# 7️⃣ Normalize features
# -----------------------------
scaler_vis = StandardScaler()
scaler_rppg = StandardScaler()
visual_features_scaled = scaler_vis.fit_transform(visual_features)
rppg_features_scaled = scaler_rppg.fit_transform(rppg_features)
early_fusion_features = np.hstack([visual_features_scaled, rppg_features_scaled])

# -----------------------------
# 8️⃣ Train-test split
# -----------------------------
X_train_ef, X_test_ef, y_train, y_test = train_test_split(
    early_fusion_features, labels_int, test_size=0.2, random_state=42)
X_train_vis, X_test_vis, _, _ = train_test_split(
    visual_features_scaled, labels_int, test_size=0.2, random_state=42)
X_train_rppg, X_test_rppg, _, _ = train_test_split(
    rppg_features_scaled, labels_int, test_size=0.2, random_state=42)

# -----------------------------
# 9️⃣ PyTorch MLP model
# -----------------------------
class MLP_PyTorch(nn.Module):
    def __init__(self, input_dim, num_classes, hidden1=128, hidden2=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(X_train, y_train, input_dim, num_classes, epochs=50, lr=0.001):
    model = MLP_PyTorch(input_dim, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    return model

# Train each model
model_vis = train_model(X_train_vis, y_train, X_train_vis.shape[1], num_classes)
model_rppg = train_model(X_train_rppg, y_train, X_train_rppg.shape[1], num_classes)
model_ef = train_model(X_train_ef, y_train, X_train_ef.shape[1], num_classes)

# -----------------------------
# 10️⃣ Evaluate models
# -----------------------------
def evaluate_model(model, X_test, y_test):
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_test_t)
        y_pred = torch.argmax(outputs, dim=1).numpy()
    acc = accuracy_score(y_test, y_pred)
    probs = torch.softmax(outputs, dim=1).numpy()
    return acc, y_pred, probs

acc_vis, y_pred_vis, prob_vis = evaluate_model(model_vis, X_test_vis, y_test)
acc_rppg, y_pred_rppg, prob_rppg = evaluate_model(model_rppg, X_test_rppg, y_test)
acc_ef, y_pred_ef, prob_ef = evaluate_model(model_ef, X_test_ef, y_test)

# Late fusion: average probabilities
prob_lf = (prob_vis + prob_rppg)/2
y_pred_lf = np.argmax(prob_lf, axis=1)
acc_lf = accuracy_score(y_test, y_pred_lf)

# -----------------------------
# 11️⃣ Save models
# -----------------------------
torch.save(model_vis.state_dict(), 'model_vis.pth')
torch.save(model_rppg.state_dict(), 'model_rppg.pth')
torch.save(model_ef.state_dict(), 'model_ef.pth')
print("All models saved as .pth files")

# -----------------------------
# 12️⃣ Ablation report
# -----------------------------
print("Task 4 Ablation Study Results:")
print(f"Visual Only Accuracy: {acc_vis*100:.2f}%")
print(f"rPPG Only Accuracy: {acc_rppg*100:.2f}%")
print(f"Early Fusion Accuracy: {acc_ef*100:.2f}%")
print(f"Late Fusion Accuracy: {acc_lf*100:.2f}%")

plt.figure(figsize=(7,4))
models = ['Visual Only','rPPG Only','Early Fusion','Late Fusion']
accuracies = [acc_vis, acc_rppg, acc_ef, acc_lf]
plt.bar(models, [a*100 for a in accuracies], color=['blue','green','orange','red'])
plt.ylabel('Accuracy (%)')
plt.title('Task 4 Ablation Study: Visual vs rPPG vs Fusion')
for i,v in enumerate([a*100 for a in accuracies]):
    plt.text(i, v+1, f"{v:.2f}%", ha='center')
plt.ylim(0,100)
plt.tight_layout()
plt.savefig('ablation_study_results.png', dpi=300)
print("Plot saved as ablation_study_results.png")
