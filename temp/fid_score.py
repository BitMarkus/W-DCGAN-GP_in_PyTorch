import os
import numpy as np
import torch
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from PIL import Image
from torchvision import transforms

"""
Input Images:

    - Ensure both folders contain only images (.png, .jpg, .jpeg).
    - Images will be auto-resized to 299x299 for Inception v3.

Interpreting FID:

    - Lower is better (0 = identical distributions).
    - FID < 50: Excellent quality.
    - FID > 200: Poor quality.
"""

# Configuration
real_images_dir = "path/to/real_images/"  # Folder with real images
fake_images_dir = "path/to/fake_images/"  # Folder with generated images
batch_size = 32                            # Reduce if GPU memory is limited
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Inception v3 for FID (remove final classification layer)
inception = inception_v3(pretrained=True, transform_input=False).to(device)
inception.eval()
inception.fc = torch.nn.Identity()  # Remove final layer to get 2048-dim features

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(299),                 # Inception v3 expects 299x299
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_features(image_folder):
    """Extract Inception v3 features for all images in a folder"""
    features = []
    img_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i+batch_size]
        batch = []
        
        for path in batch_paths:
            img = Image.open(path).convert("RGB")  # Ensure RGB (even for grayscale)
            batch.append(preprocess(img))
        
        batch = torch.stack(batch).to(device)
        with torch.no_grad():
            feat = inception(batch).cpu().numpy()
        features.append(feat)
    
    return np.concatenate(features, axis=0)

def calculate_fid(real_features, fake_features):
    """Compute FID between real and fake features"""
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Handle complex numbers due to sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

if __name__ == "__main__":
    print("Extracting features from real images...")
    real_feats = get_features(real_images_dir)
    
    print("Extracting features from fake images...")
    fake_feats = get_features(fake_images_dir)
    
    print(f"Real features: {real_feats.shape}, Fake features: {fake_feats.shape}")
    
    fid_score = calculate_fid(real_feats, fake_feats)
    print(f"FID Score: {fid_score:.2f}")