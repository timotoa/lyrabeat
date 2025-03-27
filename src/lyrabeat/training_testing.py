import torch
import time
import datetime
import numpy as np
from torch import nn
from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer


def train_network(
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: dict
) -> tuple[nn.Module, Optimizer]:
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0))

    num_epoch = config['train epochs']
    for epoch in range(1, num_epoch+1):
        time_now = datetime.datetime.now()
        time_now = time_now.strftime("%H:%M")
        desc = f'{time_now} Starting Epoch {epoch:>3}'
        for i, (spectrogram, target) in tqdm(
            enumerate(train_loader), desc=f'{desc:<25}',
            ncols=80, total=len(train_loader)
        ):
            outputs = model(spectrogram)
            loss = criterion(outputs.squeeze(1), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # flops, params = profile(model, inputs=(spectrogram,))
            # print(flops, params)
            # print(f"Flops: {flops / 1e9:.2f} billion operations")

        lr = optimizer.param_groups[0]['lr']
        error = test_network(model, criterion, test_loader, config)
        print(f'      current error: {error:.4f}, lr: {lr}\n')
        # result, error = test_network(
        # model, test_loader, config, desc='      Testing Network'
        # )
        # print(f'      current error: {error:.4f}, lr: {lr}\n')
        # if n_iter != 0 and (epoch % n_iter == 0 or epoch == num_epoch+1):
        #     prediction, error = test_network(
        #         model, pred_loader, config, desc='      Writing Predictions'
        #     )
        #     write_file(prediction, config)
    return model, optimizer


def test_network(model: nn.Module, criterion: nn.Module, dataloader: DataLoader, config: dict):
    model.eval()
    total = 0
    time_now = datetime.datetime.now()
    time_now = time_now.strftime("%H:%M")
    desc = f'{time_now} Testing Network'
    for i, (spectrogram, target) in tqdm(
        enumerate(dataloader), desc=f"{desc:<25}",
        ncols=80, total=len(dataloader)
    ):
        outputs = model(spectrogram)
        loss = criterion(outputs.squeeze(1), target)
        total += loss.sum()
    model.train()
    return total / len(dataloader.dataset)
