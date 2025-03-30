import random
import numpy as np
import torch
import torchaudio
import glob
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def log_spectrogram(waveform, sample_rate, n_fft=400, hop_length=256, n_mels=128, **kwargs):
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )(waveform)

    log_mel_spectrogram = T.AmplitudeToDB()(mel_spectrogram)
    return log_mel_spectrogram


class AudioDataset(Dataset):
    def __init__(self, dataset_path: str, annotations_path: str, config: dict):
        self.config = config
        self.device = config['device']
        self.length = config['datasize']

        self.annotations = []
        self.dataset = []
        dataset_files = os.listdir(dataset_path)[:config['debug dataset size']]
        dataset_files = set(dataset_files)
        annotations_files = set(os.listdir(annotations_path))
        for file in tqdm(dataset_files, ncols=80, desc="Load Data "):
            annotation_file = file[:-4]+".txt"
            if not file.endswith(".mp3") or annotation_file not in annotations_files:
                continue

            waveform, sr = torchaudio.load(os.path.join(dataset_path, file))
            waveform = waveform / waveform.abs().max()
            spectrogram = log_spectrogram(waveform, sr, **config)
            spectrogram = spectrogram.mean(dim=0)
            self.dataset.append(spectrogram.T)

            with open(os.path.join(annotations_path, annotation_file), 'r') as f:
                annotation_content = f.read()

            annotation = []
            for line in annotation_content.split("\n"):
                if ',' not in line:
                    break
                beat = float(line.split(',')[0])
                beat = int(beat*sr // config['hop_length'])
                annotation.append(beat)
            self.annotations.append(annotation)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        spectrogram = self.dataset[idx]
        length = self.length
        start = random.randint(0, spectrogram.size(0))
        spectrogram = spectrogram[start:start+length, :]
        spectrogram = F.pad(
            spectrogram, (0, 0, 0, length - spectrogram.size(0)), value=0
        )

        annotation = torch.zeros((length, 1))
        for beat in self.annotations[idx]:
            if start <= beat < start+length:
                annotation[beat-start] = 1
        return spectrogram.to(self.device), annotation.to(self.device)


def get_dataloaders(dataset: Dataset, config: dict) -> tuple[DataLoader]:
    batch_size = config['batch size']
    train_split = config['train split']

    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size, shuffle=False
    )

    return train_loader, test_loader
