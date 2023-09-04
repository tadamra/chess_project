import pathlib

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import EvaluationCNN
from processGame import get_samples_white

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.DoubleTensor)

lr = 1e-6
N_epochs = 14
hidden_size = 128
hidden_layers = 4
batch_size = 30
data_size = 200000
number_of_files_to_use = int(data_size/20000)
starting_file = 0

def train():

    model_for_white = EvaluationCNN.EvaluationNetwork(hidden_layers=hidden_layers, hidden_size=hidden_size)
    # model_for_white.load_state_dict(torch.load("model_white_weights.pth"))
    model_for_white = EvaluationCNN.ChessEvaluationNetwork()

    optimizer = torch.optim.Adam(model_for_white.parameters(), lr=lr)

    model_for_white.eval()
    regression_loss = nn.MSELoss()
    losses = []
    for file_number in range(number_of_files_to_use):
        targets = np.load(f"""{f"{pathlib.Path().resolve()}"[0:-18]}/data/labels{starting_file + file_number}.npy""", allow_pickle=True)
        data = np.load(f"""{f"{pathlib.Path().resolve()}"[0:-18]}/data/positions{starting_file + file_number}.npy""", allow_pickle=True)
        data_loader_train = get_samples_white(data, targets, batch_size)
        for target, sample in data_loader_train:
            with torch.no_grad():
                # sample = torch.tensor(data[0:30,:,:,:])
                # random_indices = np.random.choice(data.shape[0], size=sample_size, replace=False)
                # samples = torch.tensor(data[random_indices])
                output = model_for_white(sample).squeeze()
                loss = regression_loss(target, output)
                losses.append(loss)

    print(np.mean(losses))

    for i in tqdm(range(N_epochs)):
        model_for_white.train()
        for file_number in range(number_of_files_to_use):
            targets = np.load(f"""{f"{pathlib.Path().resolve()}"[0:-18]}/data/labels{starting_file + file_number}.npy""", allow_pickle=True)
            data = np.load(f"""{f"{pathlib.Path().resolve()}"[0:-18]}/data/positions{starting_file + file_number}.npy""", allow_pickle=True)
            data_loader_train = get_samples_white(data, targets, batch_size)
            for target, sample in data_loader_train:
                output = model_for_white(sample).squeeze()
                loss = regression_loss(target.to(torch.double), output.to(torch.double))
                # print("output",output)
                # print("target", target)
                # print(loss)
                loss.backward()
                optimizer.step()
                # output = model_for_white(sample).squeeze()
                # loss = regression_loss(target.to(torch.double), output.to(torch.double))
                # print(loss)
        torch.save(model_for_white.state_dict(), f"""model_white_weights{i+20}.pth""")
        model_for_white.eval()
        losses = []
        for file_number in range(number_of_files_to_use):
            targets = np.load(f"""{f"{pathlib.Path().resolve()}"[0:-18]}/data/labels{starting_file + file_number}.npy""", allow_pickle=True)
            data = np.load(f"""{f"{pathlib.Path().resolve()}"[0:-18]}/data/positions{starting_file + file_number}.npy""", allow_pickle=True)
            data_loader_train = get_samples_white(data, targets, batch_size)
            for target, sample in data_loader_train:
                with torch.no_grad():
                    # sample = torch.tensor(data[0:30,:,:,:])
                    # random_indices = np.random.choice(data.shape[0], size=sample_size, replace=False)
                    # samples = torch.tensor(data[random_indices])
                    output = model_for_white(sample).squeeze()
                    loss = regression_loss(target, output)
                    losses.append(loss)
        print("train loss = ", np.mean(losses))
        for file_number in range(99,100):
            targets = np.load(f"""{f"{pathlib.Path().resolve()}"[0:-18]}/data/labels{file_number}.npy""", allow_pickle=True)
            data = np.load(f"""{f"{pathlib.Path().resolve()}"[0:-18]}/data/positions{file_number}.npy""", allow_pickle=True)
            data_loader_train = get_samples_white(data, targets, batch_size)
            for target, sample in data_loader_train:
                with torch.no_grad():
                    # sample = torch.tensor(data[0:30,:,:,:])
                    # random_indices = np.random.choice(data.shape[0], size=sample_size, replace=False)
                    # samples = torch.tensor(data[random_indices])
                    output = model_for_white(sample).squeeze()
                    loss = regression_loss(target, output)
                    losses.append(loss)
        print("test loss = ", np.mean(losses))

    torch.save(model_for_white.state_dict(), 'model_white_weights1000.pth')

    return model_for_white


if __name__ == '__main__':
    targets = np.load(f"""{f"{pathlib.Path().resolve()}"[0:-18]}/data/labels2.npy""", allow_pickle=True)
    data = np.load(f"""{f"{pathlib.Path().resolve()}"[0:-18]}/data/positions2.npy""", allow_pickle=True)
    model_for_white = train()
    # model_for_white = EvaluationCNN.EvaluationNetwork(hidden_layers=hidden_layers, hidden_size=hidden_size)
    # model_for_white.load_state_dict(torch.load("model_white_weights.pth"))
    data_loader_train = get_samples_white(data, targets, batch_size)
    losses = []
    regression_loss = nn.MSELoss()
    for file_number in range(91, 100):
        targets = np.load(f"""{f"{pathlib.Path().resolve()}"[0:-18]}/data/labels{file_number}.npy""", allow_pickle=True)
        data = np.load(f"""{f"{pathlib.Path().resolve()}"[0:-18]}/data/positions{file_number}.npy""", allow_pickle=True)
        data_loader_train = get_samples_white(data, targets, batch_size)
        for target, sample in data_loader_train:
            with torch.no_grad():
                # sample = torch.tensor(data[0:30,:,:,:])
                # random_indices = np.random.choice(data.shape[0], size=sample_size, replace=False)
                # samples = torch.tensor(data[random_indices])
                output = model_for_white(sample).squeeze()
                loss = regression_loss(target, output)
                losses.append(loss)
    print("test loss = ", np.mean(losses))
