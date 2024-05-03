# import numpy as np
# import matplotlib.pyplot as plt

# def load_single_heatmap(file_path):
#     """ Load a single .npy heatmap file """
#     return np.load(file_path)
    
# def visualize_composite_heatmap(heatmaps):
#     """ Visualize a composite image made by adding up all heatmaps """
#     composite_image = np.sum(heatmaps, axis=0)  # Sum heatmaps along the channel dimension
#     composite_image = np.clip(composite_image, 0, 1)  # Clip values to [0, 1] range for display

#     plt.figure(figsize=(6, 6))
#     plt.imshow(composite_image, cmap='hot', interpolation='nearest')
#     plt.title('Composite Heatmap')
#     plt.axis('off')
#     plt.show()

    
# # Example usage:
# # Specify the path to the specific .npy heatmap file you want to visualize
# file_path = 'dataset/picking_up_from_ground/Pufg_01/00285_meng_RGB.npy'
# heatmap = load_single_heatmap(file_path)
# print(heatmap.shape)
# visualize_composite_heatmap(heatmap)


import os
import numpy as np
import matplotlib.pyplot as plt

def load_heatmap(file_path):
    """ Load a heatmap file """
    return np.load(file_path)

def visualize_heatmaps(heatmaps, channel=None):
    """ Visualize a list of heatmaps """
    n = len(heatmaps)
    fig, axs = plt.subplots(1, n, figsize=(15, 15))
    for i, heatmap in enumerate(heatmaps):
        if channel is not None:
            # Display a specific channel
            image = heatmap[:, :, channel]
        else:
            # Sum all channels to create a composite image
            image = np.sum(heatmap, axis=0)
            image = np.clip(image, 0, 1)  # Clip values to [0, 1] range for display
            
        axs[i].imshow(image, cmap='hot', interpolation='nearest')
        axs[i].set_title(f'Frame {i+1}')
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

def sample_heatmaps(directory, num_samples=5):
    """ Sample a number of .npy files from a directory and load them """
    files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.npy')]
    files.sort()  # Sort the files to maintain sequence order
    total_files = len(files)
    
    # Sample uniformly
    indices = np.linspace(0, total_files - 1, num=num_samples, dtype=int)
    sampled_files = [files[i] for i in indices]
    
    # Load heatmaps
    heatmaps = [load_heatmap(file) for file in sampled_files]
    return heatmaps

# Example usage:
directory_path = 'dataset/moving_stuff/Ms_05'
heatmaps = sample_heatmaps(directory_path)
visualize_heatmaps(heatmaps, channel=None)  # Set 'channel' to None for composite, or specify channel index (0-26)


