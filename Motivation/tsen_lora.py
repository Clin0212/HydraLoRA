import os
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import json
from safetensors.torch import load_file

# Read model paths from environment variables
model_paths = json.loads(os.getenv('MODEL_PATHS', '{}'))


# Function to load and extract model parameters
def load_and_extract_params(model_paths):
    all_samples = []
    samples_counts = []
    for path in model_paths.values():
        model_params = load_file(path, device="cpu")
        samples = [param.data.cpu().numpy().flatten() for param in model_params.values()]
        all_samples += samples
        samples_counts.append(len(samples))
    return np.array(all_samples), samples_counts

# Ensure all paths are set
if None in model_paths.values():
    raise ValueError("One or more model paths are not set. Please check your environment variables.")


# Load and extract parameters from models
all_samples, samples_counts = load_and_extract_params(model_paths)


# Apply t-SNE
tsne = TSNE(n_components=2, init='pca', random_state=42)
transformed_params = tsne.fit_transform(all_samples)

colors =['#0094C6','#F9A825','#20B2AA','#FF4E50', '#2ECC71'] #  Expanded or modified to fit the number of categories

markers = ['o', 's', '^', '*', 'D'] #  Expanded or modified to fit the number of categories

with plt.style.context(['science', 'scatter']):
    start_idx = 0
    for idx, (category, count) in enumerate(zip(model_paths.keys(), samples_counts)):
        end_idx = start_idx + count
        x = transformed_params[start_idx:end_idx, 0]
        y = transformed_params[start_idx:end_idx, 1]
        plt.scatter(x,y,
                    label=category, 
                    edgecolors='gray', 
                    facecolors=colors[idx % len(colors)],
                    marker=markers[idx % len(markers)], 
                    linewidths=0.3, 
                    s=20)
        
        for i, (x_coord, y_coord) in enumerate(zip(x, y)):
            if i < 64: 
                plt.text(x_coord, y_coord, str((start_idx + i)%64), fontsize=9) 

        start_idx = end_idx
        
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig("LoRA_weight_layer.pdf")
    plt.show()