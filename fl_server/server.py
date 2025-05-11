import flwr as fl
import torch
import numpy as np

# Define the PyTorch model
class FLModel(torch.nn.Module):
    def __init__(self):
        super(FLModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x).squeeze()

if __name__ == '__main__':
    print("Starting Flower FL Server...")
    strategy = fl.server.strategy.FedAvg()
    fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy)
