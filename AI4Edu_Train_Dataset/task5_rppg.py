import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_FILE = "/nlsasfs/home/gpucbh/vyakti13/AI4Edu_Train_Dataset/labels_train.xlsx"
VISUAL_FEATURE_DIR = "/nlsasfs/home/gpucbh/vyakti13/AI4Edu_Train_Dataset/AI4EDU_SIET_Mohit/Task1_2_Visual/visual_features_csv"
RPPG_DIR = "/nlsasfs/home/gpucbh/vyakti13/AI4Edu_Train_Dataset/AI4EDU_SIET_Mohit/Task3_rppg/rppg_outputs"

BATCH_SIZE = 8  # Slightly increased for stability
EPOCHS = 15     # Increased to allow the fusion layers to converge
NUM_CLASSES_BIN = 2
NUM_CLASSES_MULTI = 4

# =========================
# LABEL MAPPING
# =========================
def map_label_binary(label):
    return 0 if label <= 0.33 else 1

def map_label_multiclass(label):
    if label <= 0.25: return 0
    elif label <= 0.5: return 1
    elif label <= 0.75: return 2
    else: return 3

# =========================
# DATASET (With Integrity Checks)
# =========================
class MultimodalDataset(Dataset):
    def __init__(self, label_file, visual_dir, rppg_dir, binary=False):
        self.df = pd.read_excel(label_file)
        self.visual_dir = visual_dir
        self.rppg_dir = rppg_dir
        self.binary = binary
        self.valid_indices = []
        self.file_map = {} 

        v_files = os.listdir(visual_dir)
        r_files = os.listdir(rppg_dir)

        print(f"\n--- Syncing Data & Integrity Check ---")
        for i, row in self.df.iterrows():
            raw_id = row.iloc[0]
            # Handle numeric IDs or string IDs with extensions
            vid = str(int(raw_id)) if isinstance(raw_id, (int, float)) else str(raw_id).strip().split('.')[0]

            v_match = [f for f in v_files if vid in f and (f.endswith('.csv') or f.endswith('.pt'))]
            r_match = [f for f in r_files if vid in f and f.endswith('.csv')]

            if v_match and r_match:
                v_path = os.path.join(self.visual_dir, v_match[0])
                r_path = os.path.join(self.rppg_dir, r_match[0])
                
                # Check if file is empty to prevent pandas EmptyDataError
                if os.path.exists(r_path) and os.path.getsize(r_path) > 0:
                    try:
                        # Test read to ensure it's not just a header with no rows
                        test_df = pd.read_csv(r_path, nrows=1)
                        if not test_df.empty:
                            self.valid_indices.append(i)
                            self.file_map[i] = (v_path, r_path)
                    except Exception:
                        continue 

        print(f"✅ Matched {len(self.valid_indices)} valid/non-empty samples.")
        if len(self.valid_indices) == 0:
            raise ValueError("No valid data found. Check if rPPG CSVs are empty.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
        label_val = row.iloc[1]
        v_path, r_path = self.file_map[actual_idx]

        # Load Visual
        if v_path.endswith('.csv'):
            visual_feat = torch.tensor(pd.read_csv(v_path).values, dtype=torch.float32)
        else:
            visual_feat = torch.load(v_path)

        # Load rPPG
        rppg_df = pd.read_csv(r_path)
        col = 'rppg' if 'rppg' in rppg_df.columns else rppg_df.columns[-1]
        rppg_signal = torch.tensor(rppg_df[col].values, dtype=torch.float32)

        label = map_label_binary(label_val) if self.binary else map_label_multiclass(label_val)
        return visual_feat, rppg_signal, torch.tensor(label, dtype=torch.long)

# =========================
# MODEL: LATE FUSION BI-LSTM
# =========================

class MultimodalModel(nn.Module):
    def __init__(self, visual_dim=512, rppg_dim=1, hidden_visual=128, hidden_rppg=32, num_classes=2):
        super().__init__()
        # Branch 1: Visual Features
        self.visual_lstm = nn.LSTM(visual_dim, hidden_visual, batch_first=True, bidirectional=True)
        
        # Branch 2: Physiological rPPG
        self.rppg_lstm = nn.LSTM(rppg_dim, hidden_rppg, batch_first=True, bidirectional=True)
        
        # Fusion Layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_visual*2 + hidden_rppg*2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, visual, rppg):
        v_out, _ = self.visual_lstm(visual)
        v_feat = torch.mean(v_out, dim=1) # Temporal Average Pooling
        
        r_out, _ = self.rppg_lstm(rppg.unsqueeze(-1))
        r_feat = torch.mean(r_out, dim=1)
        
        fused = torch.cat([v_feat, r_feat], dim=1)
        return self.classifier(fused)

# =========================
# UTILS
# =========================
def collate_fn(batch):
    visuals, rppgs, labels = zip(*batch)
    visuals = pad_sequence(visuals, batch_first=True)
    rppgs = pad_sequence(rppgs, batch_first=True)
    return visuals, rppgs, torch.stack(labels)

def train_model(binary=True):
    dataset = MultimodalDataset(LABEL_FILE, VISUAL_FEATURE_DIR, RPPG_DIR, binary=binary)
    train_len = int(0.8 * len(dataset))
    val_set, train_set = random_split(dataset, [len(dataset)-train_len, train_len])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = MultimodalModel(num_classes=NUM_CLASSES_BIN if binary else NUM_CLASSES_MULTI).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    print(f"🚀 Training {'Binary' if binary else 'Multi-class'} Model...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for v, r, l in train_loader:
            v, r, l = v.to(DEVICE), r.to(DEVICE), l.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(v, r), l)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for v, r, l in val_loader:
            out = model(v.to(DEVICE), r.to(DEVICE))
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            targets.extend(l.numpy())
    
    print("\n📊 Results:")
    print(classification_report(targets, preds, zero_division=0))
    return model

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("=== TASK 5: MULTIMODAL ENGAGEMENT MODEL ===")
    train_model(binary=True)
    print("\n" + "="*50 + "\n")
    train_model(binary=False)