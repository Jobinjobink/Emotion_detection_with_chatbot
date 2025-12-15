import torch
import cv2
import time
import numpy as np
from torchvision import transforms
from PIL import Image

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
cnn_model = torch.load("models/resnet_fer_model.pth", map_location=device, weights_only=False)
vit_model = torch.load("models/vit_fer2013_final.pth", map_location=device, weights_only=False)

cnn_model.eval()
vit_model.eval()

CNN_WEIGHT = 0.4
VIT_WEIGHT = 0.6

cnn_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_emotion_for_5_seconds():
    cap = cv2.VideoCapture(0)
    emotion_count = {e: 0 for e in emotion_labels}

    start_time = time.time()

    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
            face_pil = Image.fromarray(face_rgb)

            cnn_input = cnn_transform(face_pil).unsqueeze(0).to(device)
            vit_input = vit_transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                cnn_out = torch.softmax(cnn_model(cnn_input), dim=1)
                vit_out = torch.softmax(vit_model(vit_input), dim=1)
                fused = CNN_WEIGHT * cnn_out + VIT_WEIGHT * vit_out
                pred = torch.argmax(fused, dim=1).item()

            emotion_count[emotion_labels[pred]] += 1

    cap.release()

    # Dominant emotion
    return max(emotion_count, key=emotion_count.get)
