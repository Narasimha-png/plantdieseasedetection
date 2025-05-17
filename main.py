import os
import torch
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import transforms
import streamlit as st
import gdown
import CNN  # Your CNN.py must be in the same directory

# Download model.pt from Google Drive if not present
model_path = "model.pt"
drive_file_id = "1UOYv_KEKdRG26z1UyfnhtMbviAUJzKGL"
gdrive_url = f"https://drive.google.com/uc?id={drive_file_id}"

if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        gdown.download(gdrive_url, model_path, quiet=False)

# Load CSV files
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load the trained model
model = CNN.CNN(39)  # Replace 39 with the correct number of output classes if different
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Image prediction function
def prediction(image_path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    transform = transforms.ToTensor()
    input_data = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
    output = model(input_data).detach().numpy()
    return np.argmax(output)

# Streamlit UI
st.title("🌿 Plant Disease Detection")
st.markdown("Upload an image of a plant leaf to detect any disease and get supplement suggestions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    os.makedirs('static/uploads', exist_ok=True)
    file_path = os.path.join('static/uploads', uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    pred = prediction(file_path)
    
    # Get prediction details
    title = disease_info['disease_name'][pred]
    description = disease_info['description'][pred]
    prevent = disease_info['Possible Steps'][pred]
    image_url = disease_info['image_url'][pred]
    supplement_name = supplement_info['supplement name'][pred]
    supplement_image_url = supplement_info['supplement image'][pred]
    supplement_buy_link = supplement_info['buy link'][pred]

    # Display prediction
    st.subheader(f"🦠 Disease: {title}")
    st.write(description)
    st.write(f"🛡️ Prevention Steps: {prevent}")
    st.image(image_url, caption="Reference Disease Image", use_column_width=True)

    st.subheader("💊 Recommended Supplement")
    st.write(supplement_name)
    st.image(supplement_image_url, caption="Supplement Image", use_column_width=True)
    st.markdown(f"[🛒 Buy Here]({supplement_buy_link})")
