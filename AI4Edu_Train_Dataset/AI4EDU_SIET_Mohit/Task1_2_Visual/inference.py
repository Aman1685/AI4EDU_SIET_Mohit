import os
import cv2
import torch
from torchvision import models, transforms
import mediapipe as mp
import torch.nn.utils.rnn as rnn_utils
from model import AttentivenessModel  # Import model class

# ====================
# DEVICE CONFIG
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================
# CLASS NAMES
# ====================
CLASS_NAMES = {0: "Disengaged", 1: "Distracted", 2: "Nominally Engaged", 3: "Highly Engaged"}

# ====================
# LOAD MODEL STATE
# ====================
model = AttentivenessModel().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
print("Model loaded successfully!")

# ====================
# FACE DETECTOR
# ====================
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

# ====================
# FEATURE EXTRACTOR
# ====================
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
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

# ====================
# VIDEO PREDICTION
# ====================
def predict_video(video_path, frame_skip=5, max_frames=60):
    cap = cv2.VideoCapture(video_path)
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
        print(f"No valid frames found in {video_path}")
        return None

    video_tensor = rnn_utils.pad_sequence(features, batch_first=True).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.argmax(model(video_tensor), dim=1).item()
    return CLASS_NAMES[pred]

# ====================
# MAIN
# ====================
if __name__ == "__main__":
    test_videos_dir = "test_videos"  # Folder with videos

    if not os.path.exists(test_videos_dir):
        print(f"Directory '{test_videos_dir}' not found!")
        exit()

    for video_file in os.listdir(test_videos_dir):
        if not video_file.lower().endswith((".mp4", ".avi", ".mov")):
            continue
        video_path = os.path.join(test_videos_dir, video_file)
        prediction = predict_video(video_path)
        print(f"Video: {video_file} --> Predicted Engagement: {prediction}")
