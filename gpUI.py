import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import io
import zipfile
import os
import tempfile
import shutil
import time  # Import time for sleep functionality

# --- Generator Model Definition ---
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# --- Helper function to load generator ---
def load_generator(checkpoint_bytes, device, z_dim=100):
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=device)
    G = Generator(z_dim).to(device)
    G.load_state_dict(checkpoint['generator_state_dict'])
    G.eval()
    return G

# --- Function to save images as a ZIP file ---
def save_images_as_zip(images, zip_filename='generated_images.zip'):
    temp_dir = tempfile.mkdtemp()
    for i, image in enumerate(images):
        img_filename = os.path.join(temp_dir, f'image_{i+1}.png')
        plt.imsave(img_filename, np.transpose(image, (1, 2, 0)))  # Save image as PNG
    # Create a ZIP file
    zip_path = os.path.join(temp_dir, zip_filename)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)
    return zip_path

# --- Streamlit App ---
st.set_page_config(page_title="🧬 MedGen - Synthetic Medical Image Generator", layout="wide")

# --- Custom Styling with CSS ---
st.markdown("""
    <style>
        body {
            background: linear-gradient(45deg, #f5f7fa, #c3cfe2);
            font-family: 'Roboto', sans-serif;
        }
        h1 {
            color: #3b3a9a;
            font-size: 3em;
            font-weight: 700;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
        }
        .sidebar .sidebar-content {
            background: linear-gradient(45deg, #d4cfcf, #a9b1d6);
            border-radius: 10px;
            padding: 15px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            font-size: 1.1em;
            transition: transform 0.3s ease-in-out;
        }
        .stButton>button:hover {
            transform: scale(1.1);
        }
        .stSpinner {
            animation: bounce 1.5s infinite;
        }
        @keyframes bounce {
            0%, 100% {
                transform: translateY(0);
            }
            50% {
                transform: translateY(-10px);
            }
        }
        .stDownloadButton>button {
            background-color: #008CBA;
            color: white;
            font-size: 1.2em;
            padding: 10px 20px;
            border-radius: 15px;
            transition: transform 0.3s ease-in-out;

        }
        
        .stDownloadButton>button:hover {
            transform: scale(1.1);
        }
        footer {
            font-size: 0.8em;
            color: #777;
            text-align: center;
            margin-top: 50px;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center;'>🧬MedGen - Synthetic Medical Image Generator</h1>", unsafe_allow_html=True)

st.sidebar.header("Upload Model Checkpoint")
uploaded_file = st.sidebar.file_uploader("Upload a .pth checkpoint", type=['pth'])

num_images = st.sidebar.slider("Number of Images to Generate", min_value=1, max_value=10, value=5)

generate_button = st.sidebar.button("Generate Images")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading animation
with st.spinner('Loading model and generating images...'):
    if uploaded_file and generate_button:
        try:
            st.success("✅ Model checkpoint uploaded successfully.")

            # Load the generator
            generator = load_generator(uploaded_file.read(), device)

            # Generate random latent vectors
            z = torch.randn(num_images, 100, 1, 1).to(device)

            # Generate images in a memory-efficient way
            generated_images = []
            for i in range(num_images):
                with torch.no_grad():
                    img = generator(z[i:i+1])  # Generate one image at a time to save memory
                img = (img + 1) / 2  # Rescale from [-1, 1] to [0, 1]
                generated_images.append(img.squeeze().cpu().numpy())

                # Add a sleep function to introduce a delay
                time.sleep(0.5)  # Adjust this value for the delay in seconds

            # Display images with padding
            grid = make_grid(torch.stack([torch.tensor(img) for img in generated_images]), 
                             nrow=min(5, num_images), normalize=True, padding=10)  # Increased padding for space
            fig, ax = plt.subplots(figsize=(12, 6))
            npimg = grid.numpy()
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.axis('off')
            st.pyplot(fig)

            # Provide option to download generated images as a ZIP file
            zip_path = save_images_as_zip(generated_images)
            with open(zip_path, 'rb') as f:
                st.download_button(
                    label="Download Generated Images as ZIP",
                    data=f,
                    file_name="generated_images.zip",
                    mime="application/zip"
                )

        except Exception as e:
            st.error(f"❌ Error loading model or generating images: {e}")
    else:
        st.info("📥 Please upload a valid .pth checkpoint file to begin.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Developed by Sam | Minor Project Year 2025 | All Rights Reserved  © tjmas04</p>", unsafe_allow_html=True)
