import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import dlib

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 7)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 2. Load Model and Set Up
device = torch.device('cpu')
model = EmotionCNN()
# Use the relative path to your model file
model.load_state_dict(torch.load("emotion_cnn_model.pth", map_location=device))
model.to(device)
model.eval() # Set the model to evaluation mode

# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


emotion_labels = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Sadness",
    5: "Surprise",
    6: "Neutral"
}


detector = dlib.get_frontal_face_detector()


cap = cv2.VideoCapture(0) 
window_name = 'Facial Emotion Detector' 

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()



while True:
    
    ret, frame = cap.read()
    if not ret:
        break

    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    faces = detector(rgb_frame)

    
    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())

        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if y < 0 or x < 0 or y+h > frame.shape[0] or x+w > frame.shape[1]:
            continue
            
        face_roi = rgb_frame[y:y+h, x:x+w]
        
        pil_image = Image.fromarray(face_roi)

        image_tensor = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            prediction_idx = predicted.item()

        display_text = emotion_labels[prediction_idx]

        cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow(window_name, frame)

    cv2.waitKey(1)
    
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("Application closed.")
