import librosa
import numpy as np

from preprocess import ZScoreNormaliser  # Import ZScoreNormaliser


class SoundGenerator:
    """SoundGenerator is responsible for generating audios from
    spectrograms using a trained VAE model.
    """

    def __init__(self, vae, hop_length, global_mean, global_std):
        self.vae = vae
        self.hop_length = hop_length
        self.global_mean = global_mean
        self.global_std = global_std
        self._normaliser = ZScoreNormaliser()

    def generate(self, spectrograms):
        # Generate reconstructed spectrograms and latent representations from the VAE
        generated_spectrograms, latent_representations = self.vae.reconstruct(spectrograms)
        # Convert spectrograms back to audio signals
        signals = self.convert_spectrograms_to_audio(generated_spectrograms)
        return signals, latent_representations

    def convert_spectrograms_to_audio(self, spectrograms):
        signals = []
        for spectrogram in spectrograms:
            # Remove the channel dimension if present
            if spectrogram.ndim == 3:
                spectrogram = spectrogram[:, :, 0]
            # Denormalize the spectrogram using the global mean and std
            denorm_log_spec = self._normaliser.denormalise(
                spectrogram, self.global_mean, self.global_std)
            # Convert from log spectrogram (dB) to linear magnitude spectrogram
            spec = librosa.db_to_amplitude(denorm_log_spec)
            # Reconstruct the time-domain signal using the Griffin-Lim algorithm
            # signal = librosa.istft(spec, hop_length=self.hop_length)
            signal = librosa.griffinlim(spec, hop_length=self.hop_length, n_iter=64)
            signal = np.nan_to_num(signal, nan=0.0)
            signal = np.nan_to_num(signal, posinf=1e6, neginf=-1e6)

            # Append the signal to the list
            signals.append(signal)
        return signals
