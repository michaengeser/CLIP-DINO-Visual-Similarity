import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, AutoImageProcessor, AutoModel
import faiss
import os
import numpy as np
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Load CLIP model and processor
processor_clip = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Load DINOv2 model and processor
processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model_dino = AutoModel.from_pretrained('facebook/dinov2-base').to(device)


# Define a function that normalizes embeddings and add them to the index
def add_vector_to_index(embedding, index):
    # convert embedding to numpy
    vector = embedding.detach().cpu().numpy()
    # Convert to float32 numpy
    vector = np.float32(vector)
    # Normalize vector: important to avoid wrong results when searching
    faiss.normalize_L2(vector)
    # Add to index
    index.add(vector)


def extract_features_clip(image):
    with torch.no_grad():
        inputs = processor_clip(images=image, return_tensors="pt").to(device)
        image_features = model_clip.get_image_features(**inputs)
        return image_features


def extract_features_dino(image):
    with torch.no_grad():
        inputs = processor_dino(images=image, return_tensors="pt").to(device)
        outputs = model_dino(**inputs)
        image_features = outputs.last_hidden_state
        return image_features.mean(dim=1)


# initialize images types 
image_types = ["own", "control"]

# loop through images types
for image_type in image_types:

    # initialize table for extracted features 
    feature_table_clip = pd.DataFrame()
    feature_table_dino = pd.DataFrame()

    # Retrieve all filenames
    images = []
    for root, dirs, files in os.walk('./' + image_type + '/'):
        for file in files:
            if file.endswith('jpg'):
                images.append(root + '/' + file)
    print('Loaded images for ', image_type)

    # Create 2 indexes.
    # index_clip = faiss.IndexFlatL2(512)
    # index_dino = faiss.IndexFlatL2(768)

    # Iterate over the dataset to extract features X2 and store features in indexes
    for image_path in images:
        img = Image.open(image_path).convert('L')
        clip_features = extract_features_clip(img)
        dino_features = extract_features_dino(img)
        # store in table
        filename = os.path.basename(image_path)
        filename = os.path.splitext(filename)[0]
        feature_table_clip[filename] = clip_features[0]
        feature_table_dino[filename] = dino_features[0]
        # add to index
        # add_vector_to_index(clip_features, index_clip)
        # add_vector_to_index(dino_features, index_dino)
        print('Extracted features for', filename)

    # Save the DataFrame to a CSV file
    clip_save_name = image_type + "_feature_table_clip.csv"
    feature_table_clip.to_csv(clip_save_name, index=False)
    dino_save_name = image_type + "_feature_table_dino.csv"
    feature_table_dino.to_csv(dino_save_name, index=False)

    # store the indexes locally
    # faiss.write_index(index_clip, "clip.index")
    # faiss.write_index(index_dino, "dino.index")

    
ws s3 sync --dryrun s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsdimagery C:/Users/JLU-SU/OneDrive - Justus-Liebig-Universität Gießen/PEP --exclude "*func1mm*" --exclude "*MNI*" --exclude "*betas_assumehrf*" --exclude "*betas_fithrf_GLMdenoise_RR*" --exclude "*betas*session*nii.gz"