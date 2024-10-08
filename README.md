# DataGeneration-VAE
Exploring Variation Auto Encoders to generate new data. From MNIST to AUDIO data generation

# Instrument Sound Generation Using Variational Autoencoders

![Project Banner](AudioGeneration_architecture.png)

Welcome to the Instrument Sound Generation project! This repository contains code and resources for generating new instrument sounds using Variational Autoencoders (VAEs). By leveraging deep learning techniques, we aim to synthesize realistic and diverse sounds of instruments like guitar, drums, violin, and piano.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Preprocessing Audio Data](#1-preprocessing-audio-data)
  - [2. Training the VAE Model](#2-training-the-vae-model)
  - [3. Generating New Sounds](#3-generating-new-sounds)
  - [4. Interactive Application](#4-interactive-application)
- [Results](#results)
- [Discussion](#discussion)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Introduction

The synthesis of new instrument sounds is a fascinating challenge in the field of audio processing and machine learning. This project explores the use of Variational Autoencoders (VAEs), a type of generative model, to learn the underlying patterns in instrument sound spectrograms and generate new, realistic sounds.

By transforming audio signals into log-scaled spectrograms and training a deep convolutional VAE, we aim to capture the essential characteristics of different instruments. The trained model can then generate novel sounds by sampling from the learned latent space.

## Dataset

We utilize the [Musical Instruments Sound Dataset](https://www.kaggle.com/datasets/soumendraprasad/musical-instruments-sound-dataset) from Kaggle, which includes:

- **Training Set**:
  - Guitar_Sound: 700 samples
  - Drum_Sound: 700 samples
  - Violin_Sound: 700 samples
  - Piano_Sound: 528 samples
- **Test Set**:
  - Total of 80 samples, 20 from each class

All audio files are in WAV format and have been standardized to a sampling rate of 22,050 Hz.

## Project Structure

```
├── dataset/
│   ├── audio_files/          # Raw audio files
│   ├── spectrograms/         # Preprocessed spectrograms
│   └── min_max_values.pkl    # Min-max normalization values
├── models/
│   └── model/                # Saved VAE model
├── samples/
│   ├── original/             # Original audio samples
│   └── generated/            # Generated audio samples
├── src/
│   ├── preprocess.py         # Audio preprocessing code
│   ├── autoencoder.py        # VAE model definition
│   ├── train.py              # Training script
│   ├── soundgenerator.py     # Sound generation utilities
│   ├── generate.py           # Audio generation script
│   ├── analysis.py           # Analysis and evaluation code
│   └── app.py                # Interactive application code
├── README.md                 # Project README
├── requirements.txt          # Python dependencies
└── LICENSE                   # License file
```

## Installation

### Prerequisites

- Python 3.7 or higher
- Git (for cloning the repository)

### Clone the Repository

```bash
git clone https://github.com/yourusername/instrument-sound-vae.git
cd instrument-sound-vae
```

### Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

*Note: Ensure that you have the necessary libraries installed, including TensorFlow, Keras, Librosa, NumPy, Matplotlib, and others specified in `requirements.txt`.*

## Usage

### 1. Preprocessing Audio Data

Transform raw audio files into log-scaled spectrograms suitable for model training.

```bash
python src/preprocess.py
```

- **Purpose**: Standardize audio data, extract features, and normalize spectrograms.
- **Output**: Preprocessed spectrograms saved in `dataset/spectrograms/` and min-max values saved in `dataset/min_max_values.pkl`.

### 2. Training the VAE Model

Train the Variational Autoencoder on the preprocessed spectrograms.

```bash
python src/train.py
```

- **Configuration**: Adjust hyperparameters like learning rate, batch size, and epochs in `train.py` if necessary.
- **Output**: Trained model saved in `models/model/` and training history logged.

### 3. Generating New Sounds

Generate new instrument sounds using the trained VAE model.

```bash
python src/generate.py
```

- **Function**: Samples latent vectors to generate new spectrograms and converts them back to audio signals.
- **Output**: Generated audio files saved in `samples/generated/`.

### 4. Interactive Application

Launch the interactive application to explore the latent space and generate sounds in real-time.

```bash
python src/app.py
```

- **Features**:
  - Visualize the latent space in 3D.
  - Click on points to generate and listen to corresponding sounds.
  - Generate new sounds by sampling from the latent space.

## Results

The VAE model effectively learned to reconstruct and generate instrument sounds:

- **Reconstruction Quality**: High fidelity in reconstructing input spectrograms, with low MSE and high SSIM scores.
- **Latent Space Organization**: Clear clustering of instrument classes in the latent space.
- **Generated Sounds**: Novel sounds that maintain the characteristic timbres of the original instruments.

Sample outputs are available in the `samples/` directory.

## Discussion

While the project achieved significant successes, certain limitations exist:

- **Dataset Diversity**: Expanding the dataset could improve generalization.
- **Model Optimization**: Further tuning of hyperparameters and architecture might enhance performance.
- **Technical Challenges**: Addressing spectrogram inversion artifacts and application responsiveness could improve results.

Future work includes exploring advanced architectures, enhancing the dataset, and optimizing the interactive application.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Dataset**: [Musical Instruments Sound Dataset](https://www.kaggle.com/datasets/soumendraprasad/musical-instruments-sound-dataset)
- **Libraries**:
  - [TensorFlow](https://www.tensorflow.org/)
  - [Keras](https://keras.io/)
  - [Librosa](https://librosa.org/)
  - [Matplotlib](https://matplotlib.org/)
  - [NumPy](https://numpy.org/)
- **Inspiration**: This project was inspired by the potential of VAEs in audio synthesis and aims to contribute to the field of machine learning-based sound generation.

---

Feel free to explore the repository and experiment with generating new instrument sounds!

If you have any questions or need assistance, please contact [your.email@example.com](mailto:your.email@example.com).

*Happy Sound Generating!*