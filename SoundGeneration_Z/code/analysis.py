import numpy as np
import matplotlib.pyplot as plt
import os
from autoencoder import VAE
from skimage.metrics import structural_similarity as ssim  # Import SSIM function

# Set the working directory if necessary
os.chdir("G:/UTS/2024/Spring_2024/Advance Data Algorithm and Machine Learning/DataGeneration-VAE/SoundGeneration_Z")

def load_InstrumentData(spectrograms_path):
    x_train = []
    X_labels = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            X_label = file_name.split('_')[0]
            spectrogram = np.load(file_path)  # (n_bins, n_frames)
            x_train.append(spectrogram)
            X_labels.append(X_label)

    x_train = np.array(x_train)
    # Add a channel dimension
    x_train = x_train[..., np.newaxis]  # -> (num_samples, n_bins, n_frames, 1)
    return x_train, X_labels

def select_images(images, labels, num_images=10):
    # Standardize labels
    standardized_labels = ["Violin" if label.lower() == "violin" else label for label in labels]
    # Filter indices based on instrument keywords
    instrument_keywords = ["Drum", "Guitar", "Piano", "Violin"]
    filtered_indices = [
        i for i, label in enumerate(standardized_labels)
        if any(keyword.lower() in label.lower() for keyword in instrument_keywords)
    ]
    # Randomly select indices from the filtered list
    sample_images_index = np.random.choice(filtered_indices, num_images, replace=False)
    # Select images and labels
    sample_images = images[sample_images_index]
    sample_labels = np.array(standardized_labels)[sample_images_index]

    return sample_images, sample_labels

def plot_reconstructed_images(images, reconstructed_images, sample_labels, mse_values, ssim_values):
    """
    Plot the original images and their reconstructed versions side by side, labeled with sample labels and evaluation metrics.
    """
    num_images = len(images)
    
    fig = plt.figure(figsize=(16, 6))
    plt.subplots_adjust(wspace=0.3, hspace=1.5)
    
    for i, (image, reconstructed_image, label, mse_value, ssim_value) in enumerate(zip(images, reconstructed_images, sample_labels, mse_values, ssim_values)):
        image = image.squeeze()  # Remove single-dimensional entries from the image
        
        # Plot the original image
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="inferno", aspect='auto')
        ax.set_title(f"Original ({label})", fontsize=10)
        
        # Plot the reconstructed image
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="inferno", aspect='auto')
        ax.set_title(f"Reconstructed\nMSE: {mse_value:.4f}\nSSIM: {ssim_value:.4f}", fontsize=10)

    plt.suptitle("Original vs Reconstructed Spectrograms", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    """
    Plot the encoded images in the latent space, colored by their labels.
    """
    # Define a color map for the instruments
    color_map = {
        'Piano': 'red',
        'Drum': 'blue',
        'Guitar': 'green',
        'Violin': 'purple'
    }
    
    # Convert labels to colors
    colors = [color_map[label] for label in sample_labels]
    
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],  # Latent dimension 1
                latent_representations[:, 1],  # Latent dimension 2
                c=colors,
                alpha=0.5,
                s=50)
    
    # Create a legend
    handles = [plt.Line2D([], [], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
               for label, color in color_map.items()]
    plt.legend(handles=handles, title="Instruments")
    
    plt.title("Latent Space Representation of Test Images", fontsize=16)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.show()

def plot_images_encoded_in_3Dlatent_space(latent_representations, sample_labels):
    """
    Plot the encoded images in the 3D latent space, colored by their labels.
    """
    # Define a color map for the instruments
    color_map = {
        'Piano': 'red',
        'Drum': 'blue',
        'Guitar': 'green',
        'Violin': 'purple'
    }
    
    # Convert labels to colors
    colors = [color_map[label] for label in sample_labels]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(latent_representations[:, 0],
               latent_representations[:, 1],
               latent_representations[:, 2],
               c=colors,
               alpha=0.5,
               s=50)
    
    # Create a legend
    handles = [plt.Line2D([], [], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
               for label, color in color_map.items()]
    ax.legend(handles=handles, title="Instruments")
    
    ax.set_title("3D Latent Space Representation of Test Images", fontsize=16)
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    ax.set_zlabel("Latent Dimension 3")
    plt.show()

def calculate_evaluation_metrics(original_images, reconstructed_images):
    """
    Calculate MSE and SSIM between original and reconstructed images.
    """
    mse_values = []
    ssim_values = []
    for original, reconstructed in zip(original_images, reconstructed_images):
        original = original.squeeze()
        reconstructed = reconstructed.squeeze()
        mse_value = np.mean((original - reconstructed) ** 2)
        mse_values.append(mse_value)
        ssim_value = ssim(original, reconstructed, data_range=original.max() - original.min())
        ssim_values.append(ssim_value)
    return mse_values, ssim_values

if __name__ == "__main__":
    SPECTROGRAMS_PATH = "dataset/spectrograms"

    autoencoder = VAE.load("model")
    x_train, X_labels = load_InstrumentData(SPECTROGRAMS_PATH)

    num_images = 10  # Adjust as needed
    sample_images, sample_labels = select_images(x_train, X_labels, num_images)

    # Reconstruct the images using the autoencoder
    reconstructed_images, latent_representations = autoencoder.reconstruct(sample_images)

    # Calculate evaluation metrics
    mse_values, ssim_values = calculate_evaluation_metrics(sample_images, reconstructed_images)

    # Plot original and reconstructed images with labels and metrics
    plot_reconstructed_images(sample_images, reconstructed_images, sample_labels, mse_values, ssim_values)

    # Proceed to plot latent space representations
    plot_images_encoded_in_latent_space(latent_representations, sample_labels)
    print("Latent space representation shape: ", latent_representations.shape)
    plot_images_encoded_in_3Dlatent_space(latent_representations, sample_labels)
