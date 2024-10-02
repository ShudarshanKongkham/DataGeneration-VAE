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
from scipy.spatial import KDTree

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
    # add a channel
    x_train = x_train[..., np.newaxis]  # -> (num_samples, n_bins, n_frames, 1)
    return x_train, X_labels


def select_images(images, labels, num_images=10):
    # Standardize all variations of "Violin" to "Violin"
    standardized_labels = ["Violin" if label.lower() == "violin" else label for label in labels]
    # Filter images based on the instrument names
    instrument_keywords = ["Drum", "Guitar", "Piano", "Violin"]
    filtered_indices = [i for i, label in enumerate(standardized_labels) if any(keyword.lower() in label.lower() for keyword in instrument_keywords)]
    # Randomly select indices from the filtered list
    sample_images_index = np.random.choice(filtered_indices, num_images, replace=False)
    # Select the images and labels based on the sampled indices
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
        # Create main frame
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create left and right frames
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Latent space plot in left frame
        self.figure = plt.Figure(figsize=(6, 6))
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.figure, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('pick_event', self.on_pick)  # Use pick_event

        # Create subframes in right_frame
        spec_frame = tk.Frame(right_frame)
        spec_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        wave_frame = tk.Frame(right_frame)
        wave_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Spectrogram plot in spec_frame
        self.spec_figure = plt.Figure(figsize=(5, 3))
        self.spec_ax = self.spec_figure.add_subplot(111)
        self.spec_canvas = FigureCanvasTkAgg(self.spec_figure, master=spec_frame)
        self.spec_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Waveform plot in wave_frame
        self.wave_figure = plt.Figure(figsize=(5, 2))
        self.wave_ax = self.wave_figure.add_subplot(111)
        self.wave_canvas = FigureCanvasTkAgg(self.wave_figure, master=wave_frame)
        self.wave_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Buttons at the bottom in a frame
        button_frame = tk.Frame(self)
        button_frame.pack(side=tk.BOTTOM, pady=10)

        self.generate_button = ttk.Button(button_frame, text="Generate New Sound", command=self.generate_new_sound)
        self.generate_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(button_frame, text="Stop Audio", command=self.stop_audio)
        self.stop_button.pack(side=tk.LEFT, padx=5)

    def load_model(self):
        # Load the VAE model
        self.vae = VAE.load("model")
        self.sound_generator = SoundGenerator(self.vae, HOP_LENGTH)
        # Load min-max values for spectrogram denormalization
        with open(MIN_MAX_VALUES_PATH, "rb") as f:
            self.min_max_values = pickle.load(f)
        # Load the data
        self.latent_representations, self.sample_labels = self.get_latent_representations()

        # Build a KDTree for nearest neighbor search
        self.kdtree = KDTree(self.latent_3d)

        # Get latent dimension
        self.latent_dim = self.vae.encoder.output_shape[-1]

    def get_latent_representations(self):
        # Load data
        x_train, X_labels = load_InstrumentData(SPECTROGRAMS_PATH)
        num_images = 28
        sample_images, sample_labels = select_images(x_train, X_labels, num_images)
        _, latent_representations = self.vae.reconstruct(sample_images)

        # Reduce dimensions for plotting using PCA
        pca3D = PCA(n_components=3)
        self.latent_3d = pca3D.fit_transform(latent_representations)

        self.latent_representations_full = latent_representations
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
        colors = [color_map[label] for label in self.sample_labels]

        scatter = self.ax.scatter(
            self.latent_3d[:, 0],
            self.latent_3d[:, 1],
            self.latent_3d[:, 2],
            c=colors,
            alpha=0.7,
            s=60,
            picker=True
        )
        self.ax.set_title("Latent Space Representation")
        # Create a legend
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color, markersize=10, label=label)
                   for label, color in color_map.items()]
        self.ax.legend(handles=handles, title="Instruments")
        self.canvas.draw()

    def generate_new_sound(self):
        # Sample a latent vector from the standard normal distribution
        latent_vector = np.random.normal(size=(1, self.latent_dim))
        # Generate spectrogram from the latent vector
        generated_spectrogram = self.vae.decoder.predict(latent_vector)
        # Remove batch dimension only
        generated_spectrogram = generated_spectrogram[0]
        # Use average min and max values for denormalization
        avg_min = np.mean([v["min"] for v in self.min_max_values.values()])
        avg_max = np.mean([v["max"] for v in self.min_max_values.values()])
        min_max_values = [{"min": avg_min, "max": avg_max}]
        # Convert spectrogram to audio
        signal = self.sound_generator.convert_spectrograms_to_audio(
            [generated_spectrogram], min_max_values
        )[0]
        # Display spectrogram and waveform, then play the audio
        self.display_and_play_audio(signal, generated_spectrogram)

    def on_pick(self, event):
        ind = event.ind
        if len(ind) > 0:
            index = ind[0]
            latent_vector = self.latent_representations_full[index].reshape(1, -1)
            # Generate spectrogram from latent vector
            generated_spectrogram = self.vae.decoder.predict(latent_vector)
            # Remove batch dimension only
            generated_spectrogram = generated_spectrogram[0]
            # Use average min and max values for denormalization
            avg_min = np.mean([v["min"] for v in self.min_max_values.values()])
            avg_max = np.mean([v["max"] for v in self.min_max_values.values()])
            min_max_values = [{"min": avg_min, "max": avg_max}]
            # Convert spectrogram to audio
            signals = self.sound_generator.convert_spectrograms_to_audio(
                [generated_spectrogram], min_max_values
            )
            signal = signals[0]
            # Display spectrogram and waveform, then play the audio
            self.display_and_play_audio(signal, generated_spectrogram)

    def display_and_play_audio(self, signal, spectrogram):
        # Extract the 2D spectrogram for plotting
        if spectrogram.ndim == 3:
            spectrogram_2d = spectrogram[:, :, 0]
        else:
            spectrogram_2d = spectrogram

        # Plot spectrogram
        self.spec_ax.clear()
        self.spec_ax.imshow(np.flipud(spectrogram_2d.T), aspect='auto', origin='lower', cmap='inferno')
        self.spec_ax.set_title('Spectrogram')
        self.spec_canvas.draw()

        # Plot waveform
        self.wave_ax.clear()
        self.wave_ax.plot(signal)
        self.wave_ax.set_title('Waveform')
        self.wave_canvas.draw()

        # Play the audio
        self.play_audio(signal)



    def play_audio(self, signal):
        self.stop_audio()  # Stop any existing playback
        sd.play(signal, samplerate=SAMPLE_RATE)

    def stop_audio(self):
        sd.stop()


if __name__ == "__main__":
    app = LatentSpaceExplorer()
    app.mainloop()
