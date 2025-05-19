import os
import torch
from PIL import Image
import numpy as np
import pandas as pd
from torchvision.transforms import ToTensor
import streamlit as st
import gdown
import CNN  # Ensure CNN.py is in the same directory

# Define model path
model_path = "model.pt"
drive_file_id = "1UOYv_KEKdRG26z1UyfnhtMbviAUJzKGL"

# Download model if not already available
if not os.path.exists(model_path):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(f"https://drive.google.com/uc?id={drive_file_id}", output=model_path, quiet=False, fuzzy=True)

# Load CSVs
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load model
model = CNN.CNN(39)  # Use correct number of classes
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

def prediction(image):
    image = image.convert("RGB").resize((224, 224))
    transform = ToTensor()
    input_data = transform(image).unsqueeze(0)
    output = model(input_data).detach().numpy()
    return np.argmax(output)

# UI
st.title("🌿 Plant Disease Detection")
st.markdown("Upload an image of a plant leaf to detect any disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image.", use_container_width=True)

    pred = prediction(image)

    title = disease_info['disease_name'][pred]
    description = disease_info['description'][pred]
    prevent = disease_info['Possible Steps'][pred]
    image_url = disease_info['image_url'][pred]
    supplement_name = supplement_info['supplement name'][pred]
    supplement_image_url = supplement_info['supplement image'][pred]
    supplement_buy_link = supplement_info['buy link'][pred]

    st.subheader(f"🦠 Disease: {title}")
    st.write(description)
    st.write(f"💡 Prevention: {prevent}")
    st.image(image_url, caption="Disease Reference Image", use_container_width=True)

    st.subheader("🧪 Recommended Supplement:")
    st.write(supplement_name)
    st.image(supplement_image_url, caption="Supplement Image", use_container_width=True)
    st.markdown(f"[🛒 Buy Here]({supplement_buy_link})")
