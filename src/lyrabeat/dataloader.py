import numpy as np
import torch
import torchaudio
import glob
import os
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchaudio.transforms as T
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
        self.annotations = []
        self.dataset = []
        dataset_files = set(os.listdir(dataset_path))
        annotations_files = set(os.listdir(annotations_path))
        i = 0
        for file in tqdm(dataset_files):
            i += 1
            if i > 3:
                break
            annotation_file = file[:-4]+".txt"
            if not file.endswith(".mp3") or annotation_file not in annotations_files:
                continue

            waveform, sr = torchaudio.load(os.path.join(dataset_path, file))
            spectrogram = log_spectrogram(waveform, sr, **config)
            spectrogram = spectrogram.mean(dim=0)

            projection = 2 ** len(config["encoder"])
            pad_size = (-spectrogram.size(1)) % projection
            spectrogram = torch.nn.functional.pad(spectrogram, (0, pad_size))

            self.dataset.append(spectrogram)

            with open(os.path.join(annotations_path, annotation_file), 'r') as f:
                annotation_content = f.read()

            annotation = torch.zeros(spectrogram.size(1))
            for line in annotation_content.split("\n")[:-1]:
                beat = float(line.split(',')[0])
                beat = int(beat*sr // config['hop_length'])
                if beat < annotation.size(0):
                    annotation[beat] = 1
            self.annotations.append(annotation)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        annotation = self.annotations[idx]
        spectrogram = self.dataset[idx]
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

# config = {
#     'n_fft': 512,
#     'hop_length': 256,
#     'n_mels': 128,
# }
# dataset_path = 'dataset_small'
# annotations_path = 'annotations'
# dataset = AudioDataset(dataset_path, annotations_path, config)
# dataset[0]
