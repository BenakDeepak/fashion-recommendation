import pickle
import numpy as np

# Load the embeddings file
with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# Find the number of embeddings
num_embeddings = embeddings.shape[0]  # Number of rows corresponds to the number of embeddings

print(f"Number of embeddings created: {num_embeddings}")
