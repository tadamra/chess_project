import pathlib
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from chess_project.neural_networks import CopiedEvalChessNetwork
from datasets import get_samples_white

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.DoubleTensor)

lr = 3e-4
N_epochs = 100
hidden_size = 128
hidden_layers = 4
sample_size = 30


def train():
    model_for_white = CopiedEvalChessNetwork(hidden_layers=hidden_layers, hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model_for_white.parameters(), lr=lr)

    model_for_white.eval()
    targets = np.load(f"""{f"{pathlib.Path().resolve()}"[0:-18]}/data/labels0.npy""", allow_pickle=True)
    data = np.load(f"""{f"{pathlib.Path().resolve()}"[0:-18]}/data/positions0.npy""", allow_pickle=True)
    data_loader_train = get_samples_white(data[0:20000, :, :, :], targets[0:20000], sample_size)
    print(targets[:,0])
    print(targets)
    regression_loss = nn.MSELoss()
    losses = []
    for target, sample in data_loader_train:

        with torch.no_grad():
            # sample = torch.tensor(data[0:30,:,:,:])
            # random_indices = np.random.choice(data.shape[0], size=sample_size, replace=False)
            # samples = torch.tensor(data[random_indices])
            output = model_for_white(sample).squeeze()
            loss = regression_loss(target, output)
            print(loss)
            losses.append(loss)
    print(np.mean(losses))

    for i in tqdm(range(N_epochs)):
        model_for_white.train()
        for target, sample in data_loader_train:
            output = model_for_white(sample).squeeze()
            loss = regression_loss(target.to(torch.double), output.to(torch.double))
            loss.backward()
            optimizer.step()

        model_for_white.eval()
        losses = []
        for target, sample in data_loader_train:
            with torch.no_grad():
                # sample = torch.tensor(data[0:30,:,:,:])
                # random_indices = np.random.choice(data.shape[0], size=sample_size, replace=False)
                # samples = torch.tensor(data[random_indices])
                output = model_for_white(sample).squeeze()
                loss = F.mse_loss(target, output)
                losses.append(loss)
        print(torch.mean(torch.tensor(losses)))

    torch.save(model_for_white.state_dict(), 'model_white_weights.pth')
    return model_for_white


if __name__ == '__main__':
    targets = np.load(f"""{f"{pathlib.Path().resolve()}"[0:-18]}/data/labels1.npy""", allow_pickle=True)
    data = np.load(f"""{f"{pathlib.Path().resolve()}"[0:-18]}/data/positions1.npy""", allow_pickle=True)
    model_for_white = train()
    data_loader_train = get_samples_white(data[2000:4000, :, :, :], targets[2000:4000, :], sample_size)
    losses = []
    regression_loss = nn.MSELoss()
    for target, sample in data_loader_train:
        with torch.no_grad():
            # sample = torch.tensor(data[0:30,:,:,:])
            # random_indices = np.random.choice(data.shape[0], size=sample_size, replace=False)
            # samples = torch.tensor(data[random_indices])
            output = model_for_white(sample).squeeze()
            loss = regression_loss(target, output)
            losses.append(loss)
    print(np.mean(losses))
