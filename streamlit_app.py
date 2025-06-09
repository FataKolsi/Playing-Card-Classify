import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import timm
from torchvision import transforms

# Define your model architecture
class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=False)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Load the model
@st.cache_resource
def load_model():
    model = SimpleCardClassifier(num_classes=53)
    model.load_state_dict(torch.load("card_classifier.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Class names based on your mapping
classes = [
    'ace of clubs', 'ace of diamonds', 'ace of hearts', 'ace of spades',
    'eight of clubs', 'eight of diamonds', 'eight of hearts', 'eight of spades',
    'five of clubs', 'five of diamonds', 'five of hearts', 'five of spades',
    'four of clubs', 'four of diamonds', 'four of hearts', 'four of spades',
    'jack of clubs', 'jack of diamonds', 'jack of hearts', 'jack of spades',
    'joker',
    'king of clubs', 'king of diamonds', 'king of hearts', 'king of spades',
    'nine of clubs', 'nine of diamonds', 'nine of hearts', 'nine of spades',
    'queen of clubs', 'queen of diamonds', 'queen of hearts', 'queen of spades',
    'seven of clubs', 'seven of diamonds', 'seven of hearts', 'seven of spades',
    'six of clubs', 'six of diamonds', 'six of hearts', 'six of spades',
    'ten of clubs', 'ten of diamonds', 'ten of hearts', 'ten of spades',
    'three of clubs', 'three of diamonds', 'three of hearts', 'three of spades',
    'two of clubs', 'two of diamonds', 'two of hearts', 'two of spades'
]

# Preprocessing (must match training time)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # Uncomment if you applied normalization during training:
    # transforms.Normalize([0.485, 0.456, 0.406],
    #                      [0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("🃏 Playing Card Classifier")
uploaded_file = st.file_uploader("Upload a playing card image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)  # <- changed here

    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # (1, C, H, W)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)

    predicted_label = classes[pred.item()]
    confidence = conf.item() * 100

    st.markdown(f"### 🧠 Prediction: `{predicted_label}`")
    st.markdown(f"### 📊 Confidence: `{confidence:.2f}%`")
