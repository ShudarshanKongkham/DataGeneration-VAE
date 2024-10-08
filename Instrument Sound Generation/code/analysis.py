import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from autoencoder import VAE
from skimage.metrics import structural_similarity as ssim  # Import SSIM function

# Set the working directory if necessary
# os.chdir("path_to_your_working_directory")

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
    Plot the original images, their reconstructed versions, and the error images side by side,
    labeled with sample labels and evaluation metrics.
    """
    num_images = len(images)
    
    fig = plt.figure(figsize=(18, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    
    for i, (image, reconstructed_image, label, mse_value, ssim_value) in enumerate(zip(images, reconstructed_images, sample_labels, mse_values, ssim_values)):
        image = image.squeeze()  # Remove single-dimensional entries from the image
        
        # Plot the original image
        ax = fig.add_subplot(3, num_images, i + 1)
        ax.axis("off")
        im1 = ax.imshow(image, cmap="inferno", aspect='auto')
        ax.set_title(f"Original ({label})", fontsize=10)
        
        # Plot the reconstructed image
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(3, num_images, i + num_images + 1)
        ax.axis("off")
        im2 = ax.imshow(reconstructed_image, cmap="inferno", aspect='auto')
        ax.set_title(f"Reconstructed\nMSE: {mse_value:.4f}\nSSIM: {ssim_value:.4f}", fontsize=10)
        
        # Plot the error image (difference between original and reconstructed)
        error_image = image - reconstructed_image
        ax = fig.add_subplot(3, num_images, i + 2 * num_images + 1)
        ax.axis("off")
        im3 = ax.imshow(error_image, cmap="bwr", aspect='auto')  # Using 'bwr' colormap to show positive and negative differences
        ax.set_title("Error \n(Orig. - Recons.)", fontsize=10)
        # Add a colorbar to the error image
        # plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Original vs Reconstructed Spectrograms and Error Images", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
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
                alpha=0.7,
                s=80)
    
    # Create a legend
    handles = [plt.Line2D([], [], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
               for label, color in color_map.items()]
    plt.legend(handles=handles, title="Instruments", fontsize=12)
    
    plt.title("Latent Space Representation of Test Images", fontsize=16)
    plt.xlabel("Latent Dimension 1", fontsize=14)
    plt.ylabel("Latent Dimension 2", fontsize=14)
    plt.grid(True)
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
    sc = ax.scatter(latent_representations[:, 0],
                    latent_representations[:, 1],
                    latent_representations[:, 2],
                    c=colors,
                    alpha=0.7,
                    s=80)
    
    # Create a legend
    handles = [plt.Line2D([], [], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
               for label, color in color_map.items()]
    ax.legend(handles=handles, title="Instruments", fontsize=12)
    
    ax.set_title("3D Latent Space Representation of Test Images", fontsize=16)
    ax.set_xlabel("Latent Dimension 1", fontsize=12)
    ax.set_ylabel("Latent Dimension 2", fontsize=12)
    ax.set_zlabel("Latent Dimension 3", fontsize=12)
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
        # Clip values to [0, 1] if necessary for SSIM calculation
        original_clipped = np.clip(original, 0, 1)
        reconstructed_clipped = np.clip(reconstructed, 0, 1)
        ssim_value = ssim(original_clipped, reconstructed_clipped, data_range=original_clipped.max() - original_clipped.min())
        ssim_values.append(ssim_value)
    return mse_values, ssim_values

def plot_LossCurve(log_filepath):
    training_history = pd.read_csv(log_filepath)
    # Extract data for the three loss components
    epochs = range(len(training_history))
    loss = training_history['loss']
    reconstruction_loss = training_history['calculate_reconstruction_loss']
    kl_loss = training_history['calculate_kl_loss']
    # Plot the KL Loss and Reconstruction Loss separately
    plt.figure(figsize=(10, 6))

    # Plot KL loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, kl_loss, color='red')
    plt.title('KL Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('KL Loss')

    # Plot Reconstruction loss
    plt.subplot(2, 1, 2)
    plt.plot(epochs, reconstruction_loss, color='orange')
    plt.title('Reconstruction Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction Loss')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    SPECTROGRAMS_PATH = "dataset/spectrograms"
    HISTORY_LOG_PATH = "training_history.csv"

    # Load the pre-trained VAE model
    autoencoder = VAE.load("model")
    x_train, X_labels = load_InstrumentData(SPECTROGRAMS_PATH)

    num_images = 8  # Adjust as needed
    sample_images, sample_labels = select_images(x_train, X_labels, num_images)


    # Reconstruct the images using the autoencoder
    reconstructed_images, latent_representations = autoencoder.reconstruct(sample_images)

    plot_LossCurve(HISTORY_LOG_PATH)

    # Calculate evaluation metrics
    mse_values, ssim_values = calculate_evaluation_metrics(sample_images, reconstructed_images)

    # Plot original, reconstructed, and error images with labels and metrics
    plot_reconstructed_images(sample_images, reconstructed_images, sample_labels, mse_values, ssim_values)

    # Proceed to plot latent space representations
    if latent_representations.shape[1] >= 2:
        plot_images_encoded_in_latent_space(latent_representations, sample_labels)
    else:
        print("Latent space has less than 2 dimensions. Cannot plot 2D latent space.")

    if latent_representations.shape[1] >= 3:
        plot_images_encoded_in_3Dlatent_space(latent_representations, sample_labels)
    else:
        print("Latent space has less than 3 dimensions. Cannot plot 3D latent space.")

    