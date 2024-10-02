import os
import pickle
import librosa
import numpy as np

os.chdir("G:/UTS/2024/Spring_2024/Advance Data Algorithm and Machine Learning/DataGeneration-VAE/SoundGeneration_Z")


class Loader:
    """Loader is responsible for loading an audio file."""

    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration  # Duration to load
        self.mono = mono

    def load(self, file_path, offset=0):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono,
                              offset=offset)[0]
        return signal


class Padder:
    """Padder is responsible to apply padding to an array."""

    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (num_missing_items, 0),
                              mode=self.mode)
        return padded_array

    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (0, num_missing_items),
                              mode=self.mode)
        return padded_array


class LogSpectrogramExtractor:
    """LogSpectrogramExtractor extracts log spectrograms (in dB) from a
    time-series signal.
    """

    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_length)[:-1]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram


class ZScoreNormaliser:
    """ZScoreNormaliser applies z-score normalization to an array."""

    def normalise(self, array, mean, std):
        norm_array = (array - mean) / std
        return norm_array

    def denormalise(self, norm_array, mean, std):
        array = norm_array * std + mean
        return array


class Saver:
    """Saver is responsible to save features and the global mean and std."""

    def __init__(self, feature_save_dir, mean_std_save_dir):
        self.feature_save_dir = feature_save_dir
        self.mean_std_save_dir = mean_std_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)
        return save_path

    def save_global_mean_std(self, mean, std):
        save_path = os.path.join(self.mean_std_save_dir, "global_mean_std.pkl")
        data = {'mean': mean, 'std': std}
        self._save(data, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path


class PreprocessingPipeline:
    """PreprocessingPipeline processes audio files in a directory, applying
    the following steps to each file:
        1- Load a file
        2- Pad the signal (if necessary)
        3- Extract log spectrogram from signal
        4- Collect features to compute global mean and std
        5- Normalize spectrograms using z-score normalization
        6- Save the normalized spectrograms

    Storing the global mean and std values for all the log spectrograms.
    """

    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self._loader = None
        self._num_expected_samples = None
        self.features = []
        self.file_paths = []

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process(self, audio_files_dir):
        # First pass: extract features
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f"Processed file {file_path}")

        # Second pass: compute global mean and std
        self.features = np.array(self.features)
        global_mean = np.mean(self.features)
        global_std = np.std(self.features)
        print(f"Global mean: {global_mean}, Global std: {global_std}")

        # Save the global mean and std
        self.saver.save_global_mean_std(global_mean, global_std)

        # Third pass: normalize and save features
        for feature, file_path in zip(self.features, self.file_paths):
            norm_feature = self.normaliser.normalise(feature, global_mean, global_std)
            save_path = self.saver.save_feature(norm_feature, file_path)
            print(f"Saved normalized feature to {save_path}")

    def _process_file(self, file_path):
        total_duration = self._get_audio_duration(file_path)
        if total_duration >= 7 + self.loader.duration:
            offset = 7
        elif total_duration >= self.loader.duration:
            offset = 0
        else:
            offset = 0
        signal = self.loader.load(file_path, offset=offset)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        self.features.append(feature)
        self.file_paths.append(file_path)

    def _get_audio_duration(self, file_path):
        return librosa.get_duration(filename=file_path)

    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal


if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 5.936  # in seconds
    SAMPLE_RATE = 22050
    MONO = True

    SPECTROGRAMS_SAVE_DIR = "dataset/spectrograms/"
    MEAN_STD_SAVE_DIR = "dataset/"
    FILES_DIR = "dataset/Train_submission/"

    # Instantiate all objects
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    zscore_normaliser = ZScoreNormaliser()
    saver = Saver(SPECTROGRAMS_SAVE_DIR, MEAN_STD_SAVE_DIR)

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normaliser = zscore_normaliser
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.process(FILES_DIR)
