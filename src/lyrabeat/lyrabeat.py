import yaml
import torch

from .dataloader import AudioDataset, get_dataloaders
from .training_testing import train_network
from .model import AudioTransformer


def run(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('meta')
    config['device'] = device

    dataset_path = config["dataset path"]
    annotations_path = config["annotations path"]

    dataset = AudioDataset(
        dataset_path, annotations_path, config
    )
    train_loader, test_loader = get_dataloaders(dataset, config)

    model = AudioTransformer(config)
    model.to(device)

    lr = config['learning rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model, optimizer = train_network(
        model, optimizer, train_loader, test_loader, config
    )


def main():
    config_path = 'config.yaml'
    run(config_path)


if __name__ == '__main__':
    main()
