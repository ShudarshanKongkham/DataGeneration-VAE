import os
import pickle
import numpy as np
import soundfile as sf
from pathlib import Path
from soundgenerator import SoundGenerator
from autoencoder import VAE
from train import SPECTROGRAMS_PATH

# Adjust the working directory if necessary
# os.chdir("G:/UTS/2024/Spring_2024/Advance Data Algorithm and Machine Learning/DataGeneration-VAE/Instrument Sound Generation")

HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = "samples/original/"
SAVE_DIR_GENERATED = "samples/generated/"
MEAN_STD_VALUES_PATH = "dataset/global_mean_std.pkl"

def find_max_shape(spectrograms_path):
    max_rows, max_cols = 0, 0
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)
            rows, cols = spectrogram.shape
            max_rows = max(max_rows, rows)
            max_cols = max(max_cols, cols)
    return max_rows, max_cols

def load_InstrumentData(spectrograms_path):
    x_train = []
    file_paths = []
    max_rows, max_cols = find_max_shape(spectrograms_path)
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)  # (n_bins, n_frames)
            # Pad or trim spectrogram
            padded_spectrogram = np.zeros((max_rows, max_cols))
            rows, cols = spectrogram.shape
            padded_spectrogram[:rows, :cols] = spectrogram[:max_rows, :max_cols]
            x_train.append(padded_spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]  # Add channel dimension
    return x_train, file_paths

def select_spectrograms(spectrograms, file_paths, num_spectrograms=2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms, replace=False)
    sampled_spectrograms = spectrograms[sampled_indexes]
    sampled_file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_file_paths = [str(Path(fp).as_posix()) for fp in sampled_file_paths]
    return sampled_spectrograms, sampled_file_paths

def save_signals(signals, save_dir, filenames, sample_rate=22050):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for signal, name in zip(signals, filenames):
        save_path = os.path.join(save_dir, f"{name[len('dataset/spectrograms/'):-4]}.wav")
        sf.write(save_path, signal, sample_rate)

if __name__ == "__main__":
    # Initialize sound generator
    vae = VAE.load("model")

    # Load global mean and std
    with open(MEAN_STD_VALUES_PATH, "rb") as f:
        mean_std_values = pickle.load(f)
        global_mean = mean_std_values['mean']
        global_std = mean_std_values['std']

    sound_generator = SoundGenerator(vae, HOP_LENGTH, global_mean, global_std)

    # Load spectrograms and file paths
    specs, file_paths = load_InstrumentData(SPECTROGRAMS_PATH)

    # Sample spectrograms
    sampled_specs, sampled_file_paths = select_spectrograms(
        specs, file_paths, num_spectrograms=5
    )

    # Generate audio for sampled spectrograms
    signals, _ = sound_generator.generate(sampled_specs)

    # Convert original spectrogram samples to audio
    original_signals = sound_generator.convert_spectrograms_to_audio(
        sampled_specs
    )

    # Save audio signals
    save_signals(signals, SAVE_DIR_GENERATED, sampled_file_paths)

    save_signals(original_signals, SAVE_DIR_ORIGINAL, sampled_file_paths)
    print("AUDIO Generated!")
