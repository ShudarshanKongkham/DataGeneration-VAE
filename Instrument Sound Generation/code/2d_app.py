import tkinter as tk
from tkinter import ttk
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import sounddevice as sd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import cKDTree  # Use cKDTree for efficiency

from autoencoder import VAE
from soundgenerator import SoundGenerator

# Adjust imports and paths according to your project structure
SPECTROGRAMS_PATH = "dataset/spectrograms"
MIN_MAX_VALUES_PATH = "dataset/min_max_values.pkl"
HOP_LENGTH = 256
SAMPLE_RATE = 22050

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

def select_images(images, labels, num_images=100):
    # Standardize labels
    standardized_labels = [label.capitalize() for label in labels]
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

class LatentSpaceExplorer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Latent Space Explorer")
        self.geometry("1200x800")
        self.create_widgets()
        self.load_model()
        self.plot_latent_space()

    def create_widgets(self):
        # Create frames for layout
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_frame = ttk.Frame(self.control_frame)
        self.info_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a matplotlib figure for latent space
        self.figure = plt.Figure(figsize=(6,6))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Create a figure for waveform and spectrogram
        self.figure_ws = plt.Figure(figsize=(5,6))
        self.ax_waveform = self.figure_ws.add_subplot(211)
        self.ax_spectrogram = self.figure_ws.add_subplot(212)
        self.canvas_ws = FigureCanvasTkAgg(self.figure_ws, master=self.info_frame)
        self.canvas_ws.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Buttons for control
        self.generate_button = ttk.Button(self.control_frame, text="Generate Random Sound", command=self.generate_random_sound)
        self.generate_button.pack(side=tk.TOP, pady=10, padx=10)
        self.stop_button = ttk.Button(self.control_frame, text="Stop Audio", command=self.stop_audio)
        self.stop_button.pack(side=tk.TOP, pady=10, padx=10)

    def load_model(self):
        # Load the VAE model
        self.vae = VAE.load("model")
        # Determine latent dimension
        self.latent_dim = self.vae.encoder.output_shape[-1]
        self.sound_generator = SoundGenerator(self.vae, HOP_LENGTH)
        # Load min-max values for spectrogram denormalization
        with open(MIN_MAX_VALUES_PATH, "rb") as f:
            self.min_max_values = pickle.load(f)
        # Load the data and get latent representations
        self.latent_representations, self.sample_labels = self.get_latent_representations()
        # Build a KDTree for nearest neighbor search (if needed)
        self.kdtree = cKDTree(self.latent_2d)

    def get_latent_representations(self):
        # Load data
        x_train, X_labels = load_InstrumentData(SPECTROGRAMS_PATH)
        num_images = 28  # Increase for better visualization
        sample_images, sample_labels = select_images(x_train, X_labels, num_images)
        # Get latent representations
        _, latent_representations = self.vae.reconstruct(sample_images)

        # Reduce dimensions for plotting (using PCA)
        self.pca = PCA(n_components=2)
        self.latent_2d = self.pca.fit_transform(latent_representations)

        return latent_representations, sample_labels

    def plot_latent_space(self):
        # Define a color map for the instruments
        color_map = {
            'Piano': 'red',
            'Drum': 'blue',
            'Guitar': 'green',
            'Violin': 'purple'
        }

        # Convert labels to colors
        colors = [color_map.get(label, 'black') for label in self.sample_labels]

        self.ax.clear()
        scatter = self.ax.scatter(
            self.latent_2d[:, 0],
            self.latent_2d[:, 1],
            c=colors,
            alpha=0.6,
            s=50,
            picker=True
        )
        self.ax.set_title("Latent Space Representation")
        self.ax.set_xlabel("Latent Dimension 1")
        self.ax.set_ylabel("Latent Dimension 2")

        # Create a legend
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color, markersize=10, label=label)
                   for label, color in color_map.items()]
        self.ax.legend(handles=handles, title="Instruments", loc='best')
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes == self.ax:
            x = event.xdata
            y = event.ydata
            print(f"Clicked at ({x:.2f}, {y:.2f})")
            # Mark the clicked point
            self.ax.plot(x, y, 'kx', markersize=12, markeredgewidth=2)
            self.canvas.draw()
            # Generate and display audio
            self.generate_audio([x, y])

    def generate_audio(self, clicked_point):
        # Convert clicked point back to the original latent space dimensions
        latent_vector_2d = np.array(clicked_point).reshape(1, -1)
        # Inverse PCA transformation
        latent_vector = self.pca.inverse_transform(latent_vector_2d)
        # Generate spectrogram from latent vector
        generated_spectrogram = self.vae.decoder.predict(latent_vector)
        # Use average min and max values for denormalization
        avg_min = np.mean([v["min"] for v in self.min_max_values.values()])
        avg_max = np.mean([v["max"] for v in self.min_max_values.values()])
        min_max_values = [{"min": avg_min, "max": avg_max}]
        # Convert spectrogram to audio
        signals = self.sound_generator.convert_spectrograms_to_audio(
            generated_spectrogram, min_max_values
        )
        signal = signals[0]
        # Play the audio
        self.play_audio(signal)
        # Visualize waveform and spectrogram
        self.display_waveform_and_spectrogram(signal, generated_spectrogram[0])

    def generate_random_sound(self):
        # Sample a latent vector from the standard normal distribution
        latent_vector = np.random.normal(size=(1, self.latent_dim))
        # Generate spectrogram from the latent vector
        generated_spectrogram = self.vae.decoder.predict(latent_vector)
        # Use average min and max values for denormalization
        avg_min = np.mean([v["min"] for v in self.min_max_values.values()])
        avg_max = np.mean([v["max"] for v in self.min_max_values.values()])
        min_max_values = [{"min": avg_min, "max": avg_max}]
        # Convert spectrogram to audio
        signal = self.sound_generator.convert_spectrograms_to_audio(generated_spectrogram, min_max_values)[0]
        # Play the audio
        self.play_audio(signal)
        # Visualize waveform and spectrogram
        self.display_waveform_and_spectrogram(signal, generated_spectrogram[0])

    def display_waveform_and_spectrogram(self, signal, spectrogram):
        # Clear previous plots
        self.ax_waveform.clear()
        self.ax_spectrogram.clear()

        # Plot waveform
        times = np.arange(len(signal)) / SAMPLE_RATE
        self.ax_waveform.plot(times, signal)
        self.ax_waveform.set_title("Waveform")
        self.ax_waveform.set_xlabel("Time [s]")
        self.ax_waveform.set_ylabel("Amplitude")

        # Plot spectrogram
        spectrogram = spectrogram.squeeze()
        self.ax_spectrogram.imshow(spectrogram.T, origin='lower', aspect='auto', cmap='inferno')
        self.ax_spectrogram.set_title("Spectrogram")
        self.ax_spectrogram.set_xlabel("Time Frames")
        self.ax_spectrogram.set_ylabel("Frequency Bins")

        # Redraw the canvas
        self.canvas_ws.draw()

    def play_audio(self, signal):
        self.stop_audio()  # Stop any existing playback
        sd.play(signal, samplerate=SAMPLE_RATE)

    def stop_audio(self):
        sd.stop()

if __name__ == "__main__":
    app = LatentSpaceExplorer()
    app.mainloop()
