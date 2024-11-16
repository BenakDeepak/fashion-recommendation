import os
import numpy as np
import pandas as pd
import pickle
import requests
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.preprocessing import normalize

# File paths
EXCEL_FILE = r'C:\Users\Deepak\Desktop\projects\fashion-recommendation\DATASET.xlsx'
EMBEDDINGS_FILE = 'embeddings.pkl'
FILENAMES_FILE = 'filenames.pkl'
PROCESSED_IMAGES_FILE = 'processed_images.pkl'

# Load ResNet50 model with pre-trained weights, excluding top layers
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# Function to extract features from an image URL
def extract_features_from_url(image_url):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        return features.flatten()
    except Exception as e:
        print(f"Error processing {image_url}: {e}")
        return None

# Load processed images
def load_processed_images():
    if os.path.exists(PROCESSED_IMAGES_FILE):
        with open(PROCESSED_IMAGES_FILE, 'rb') as f:
            return set(pickle.load(f))
    return set()

# Save processed images
def save_processed_images(processed_images):
    with open(PROCESSED_IMAGES_FILE, 'wb') as f:
        pickle.dump(list(processed_images), f)

# Load embeddings and filenames
def load_embeddings_and_filenames():
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(FILENAMES_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            embeddings = pickle.load(f)
        with open(FILENAMES_FILE, 'rb') as f:
            filenames = pickle.load(f)
    else:
        embeddings = np.empty((0, 2048))  # Assuming 2048 is the feature size
        filenames = []
    return embeddings, filenames

# Save updated embeddings and filenames
def save_embeddings_and_filenames(embeddings, filenames):
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings, f)
    with open(FILENAMES_FILE, 'wb') as f:
        pickle.dump(filenames, f)

# Function to delete an image and its embedding
def delete_embedding(image_name):
    embeddings, filenames = load_embeddings_and_filenames()
    
    # Find the index of the image in the filenames list
    try:
        image_index = filenames.index(image_name)
    except ValueError:
        print(f"Image '{image_name}' not found in filenames.")
        return
    
    # Remove the corresponding embedding and filename
    embeddings = np.delete(embeddings, image_index, axis=0)
    filenames.pop(image_index)

    # Save the updated embeddings and filenames
    save_embeddings_and_filenames(embeddings, filenames)
    
    print(f"Deleted embedding and filename for image: {image_name}")

# Function to update embeddings from the dataset
def update_embeddings():
    processed_images = load_processed_images()

    # Load existing embeddings and filenames
    embeddings, filenames = load_embeddings_and_filenames()

    new_embeddings = []
    new_filenames = []

    # Convert old_filenames to a set for quick lookup
    existing_filenames = set(filenames)

    # Load the dataset from Excel file
    df = pd.read_excel(EXCEL_FILE)

    # Rename columns if needed
    df.rename(columns={"URL'S": 'URL', 'TYPE': 'Dress Type'}, inplace=True)

    # Iterate over the dataset to process new images
    for idx, row in df.iterrows():
        image_url = row['URL']
        dress_type = row['Dress Type']
        
        # Create a unique identifier for the image
        relative_filename = f"{dress_type}_{idx}"
        
        if relative_filename not in processed_images and relative_filename not in existing_filenames:
            features = extract_features_from_url(image_url)
            
            if features is not None:
                new_filenames.append(relative_filename)
                new_embeddings.append(features)
                processed_images.add(relative_filename)

    # If there are new embeddings, update the files
    if new_embeddings:
        new_embeddings = np.array(new_embeddings)
        new_embeddings = normalize(new_embeddings)  # Normalize embeddings

        # Combine old and new embeddings without reprocessing old files
        all_embeddings = np.vstack([embeddings, new_embeddings])
        all_filenames = filenames + new_filenames

        # Save the updated embeddings and filenames
        save_embeddings_and_filenames(all_embeddings, all_filenames)
        save_processed_images(processed_images)

if __name__ == "__main__":
    # Example usage of deleting an image and its embedding
    image_name_to_delete = "saree_0"  # Replace with the name of the image to be deleted
    delete_embedding(image_name_to_delete)
    
    # Update embeddings when new images are added
    update_embeddings()
