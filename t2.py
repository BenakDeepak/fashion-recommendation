import streamlit as st
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests
from io import BytesIO
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import Model # type: ignore
import pandas as pd

# Load embeddings and filenames
with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)
with open('filenames.pkl', 'rb') as f:
    filenames = pickle.load(f)

# Load dataset with URLs and Dress Types
EXCEL_FILE = './DATASET.xlsx'  # Update with the correct path
df = pd.read_excel(EXCEL_FILE)
df.rename(columns={"URL'S": 'URL', 'TYPE': 'Dress Type'}, inplace=True)

# Manually specify clothing types
clothing_types = df['Dress Type'].unique()

# Define the size for resizing images
IMAGE_SIZE = (224, 224)

# Function to extract features from an uploaded image
def extract_features_from_image(img):
    try:
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        model = Model(inputs=base_model.input, outputs=base_model.output)

        img = img.resize(IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)

        return features.flatten()
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
        return None

# Function to resize an image to the specified size
def resize_image(img, size):
    return img.resize(size)

# Streamlit app
st.set_page_config(page_title="Fashion Recommendation System", layout="wide")
st.title('Fashion Recommendation System')
st.markdown("### Input your fashion image URL and find similar styles!")

# Dropdown for clothing type selection
selected_clothing_type = st.selectbox("Select Clothing Type", clothing_types)

# Input for image URL
image_url = st.text_input("Enter Image URL")

if image_url:
    try:
        # Load image from URL
        response = requests.get(image_url)
        uploaded_image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Resize the uploaded image
        resized_uploaded_image = resize_image(uploaded_image, IMAGE_SIZE)
        st.image(resized_uploaded_image, caption='Uploaded Image', use_column_width=True)

        if st.button('Find Similar Images'):
            # Extract features from the uploaded image
            uploaded_image_features = extract_features_from_image(uploaded_image)
            
            if uploaded_image_features is not None:
                uploaded_image_features = uploaded_image_features.reshape(1, -1)
                
                # Compute similarities and find top indices
                similarities = cosine_similarity(uploaded_image_features, embeddings).flatten()
                top_indices = similarities.argsort()[::-1]  # Sort in descending order
                
                # Retrieve filenames of top similar images
                similar_images = [filenames[i] for i in top_indices]
                
                # Ensure the uploaded image itself is not included in the recommendations
                uploaded_file_name = image_url.split("/")[-1].lower()
                similar_images = [img for img in similar_images if os.path.basename(img).lower() != uploaded_file_name]
                
                # Filter images by selected clothing type
                filtered_similar_images = [img for img in similar_images if img.startswith(selected_clothing_type)]
                
                # Ensure exactly 3 images are shown
                if len(filtered_similar_images) < 3:
                    # If fewer than 3 similar images in the selected clothing type, add more images from the folder
                    additional_images = [img for img in filenames if img.startswith(selected_clothing_type) and os.path.basename(img).lower() != uploaded_file_name]
                    filtered_similar_images.extend(additional_images[:3 - len(filtered_similar_images)])
                
                # If fewer than 3 images are available after filtering, fill with most similar images from all categories
                if len(filtered_similar_images) < 3:
                    additional_images = [img for img in filenames if os.path.basename(img).lower() != uploaded_file_name]
                    filtered_similar_images.extend(additional_images[:3 - len(filtered_similar_images)])
                
                # Display URLs of the similar images from the dataset
                st.write('### Similar Image URLs:')
                for img_file in filtered_similar_images[1:5]:
                    # Find the matching URL from the dataset
                    matching_url = df[df['Dress Type'] == selected_clothing_type]['URL'].iloc[filenames.index(img_file)]
                    st.write(matching_url)

    except Exception as e:
        st.error(f"An error occurred while fetching the image from the URL: {e}")

# Add some CSS for better styling
st.markdown("""
    <style>
        .streamlit-expanderHeader {
            font-size: 24px;
            color: #4CAF50;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
        }
    </style>
""", unsafe_allow_html=True)
