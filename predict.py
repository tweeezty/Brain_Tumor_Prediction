import torch
import cv2
from torchvision import transforms
from PIL import Image
from model import BrainTumorCNN

# Load trained model
model = BrainTumorCNN()
model.load_state_dict(torch.load("brain_tumor_model.pth", map_location=torch.device('cpu')))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Read image
img = cv2.imread("dataset/train/no_tumor/Tr-no_0015.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img)
img = transform(img).unsqueeze(0)

# Prediction
with torch.no_grad():
    output = model(img)

if output.item() > 0.5:
    print("🧠 Brain Tumor Detected")
else:
    print("✅ No Brain Tumor")
