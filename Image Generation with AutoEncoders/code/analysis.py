import numpy as np
import matplotlib.pyplot as plt

# Import Autoencoder class and the function to load MNIST data
from autoencoder import Autoencoder
from train import load_mnist


# Function to select a specified number of random images and their corresponding labels
def select_images(images, labels, num_images=10):
    """
    Select a random sample of images and their labels from the dataset.
    
    Parameters:
        images (numpy.ndarray): The images from which to sample.
        labels (numpy.ndarray): The corresponding labels for the images.
        num_images (int): Number of images to sample.

    Returns:
        sample_images (numpy.ndarray): A set of randomly selected images.
        sample_labels (numpy.ndarray): The labels corresponding to the selected images.
    """
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels


# Function to plot original and reconstructed images side by side
def plot_reconstructed_images(images, reconstructed_images):
    """
    Plot the original images and their reconstructed versions side by side.

    Parameters:
        images (numpy.ndarray): The original images.
        reconstructed_images (numpy.ndarray): The reconstructed images by the autoencoder.
    """
    num_images = len(images)
    
    # Increase the figure size for more space between images
    fig = plt.figure(figsize=(16, 4))  
    
    # Adjust layout to add space between the rows and titles
    plt.subplots_adjust(wspace=0.3, hspace=0.2)  
    
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()  # Remove single-dimensional entries from the image
        
        # Plot the original image
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        ax.set_title("Original Image", fontsize=10)  # Smaller font size to avoid crowding
        
        # Plot the reconstructed image
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
        ax.set_title("\n Reconstructed Image", fontsize=10)  # Smaller font size for consistency

    # Set an overall title for the figure and adjust its position
    plt.suptitle("Original vs Reconstructed Images", fontsize=18)  # Move the title higher
    # Use tight_layout to automatically adjust subplot spacing
    plt.tight_layout()

   
    plt.subplots_adjust(top=0.85)  # Leave enough space for the suptitle

    plt.show()




# Function to plot the latent space representation of images
def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    """
    Plot the encoded images in the latent space, colored by their labels.
    
    Parameters:
        latent_representations (numpy.ndarray): The encoded representations in latent space.
        sample_labels (numpy.ndarray): The labels corresponding to the encoded images.
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],  # Latent dimension 1
                latent_representations[:, 1],  # Latent dimension 2
                cmap="rainbow",  # Color map for the points
                c=sample_labels,  # Color the points based on their labels
                alpha=0.5,  # Transparency of points
                s=2)  # Size of points
    plt.colorbar()  # Add a color bar to show the class-color relationship
    plt.title("Latent Space Representation of Test Images", fontsize=16)  # Plot title
    plt.xlabel("Latent Dimension 1")  # Label for x-axis
    plt.ylabel("Latent Dimension 2")  # Label for y-axis
    plt.show()


# Main section where the autoencoder is loaded, and images are processed and plotted
if __name__ == "__main__":
    # Load a pre-trained autoencoder model
    autoencoder = Autoencoder.load("model")

    # Load the MNIST dataset (train and test sets)
    x_train, y_train, x_test, y_test = load_mnist()

    # Select a small sample of images to show the original and reconstructed versions
    num_sample_images_to_show = 8
    sample_images, _ = select_images(x_test, y_test, num_sample_images_to_show)
    
    # Reconstruct the images using the autoencoder
    reconstructed_images, _ = autoencoder.reconstruct(sample_images)
    
    # Plot the original and reconstructed images
    plot_reconstructed_images(sample_images, reconstructed_images)

    # Select a large sample of images for plotting in the latent space
    num_images = 9999
    sample_images, sample_labels = select_images(x_test, y_test, num_images)
    
    # Get the latent representations of the sample images
    _, latent_representations = autoencoder.reconstruct(sample_images)
    
    # Plot the encoded images in the latent space
    plot_images_encoded_in_latent_space(latent_representations, sample_labels)
