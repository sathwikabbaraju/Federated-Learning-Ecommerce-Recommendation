import os
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# Set server address dynamically
SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "172.20.0.2:8080")
# Define dataset path inside container
CLIENT_NAME = os.getenv("CLIENT_NAME", "fl_client_1_ratermax")
data_path = f"/app/data/products_{CLIENT_NAME.split('_')[-1]}.csv"

# Load dataset and normalize features
df = pd.read_csv(data_path)
print(f"[{CLIENT_NAME}] Loaded dataset with {len(df)} records.")

# Normalize input features
feature_cols = ["rating", "no_of_ratings", "discount_price", "actual_price"]
df[feature_cols] = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()
df = df.dropna(subset=feature_cols + ['purchased'])

class RecommendationDataset(Dataset):
    def __init__(self, dataframe):
        self.X = torch.tensor(dataframe[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(dataframe["purchased"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLPModel(nn.Module):
    def __init__(self, input_dim=4):
        super(MLPModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x).squeeze()

# Training function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    for batch_idx, (features, label) in enumerate(dataloader):
        features, label = features.to(device), label.to(device)

        # Debugging Prints
        print(f"Features Tensor Shape: {features.shape}")
        print(f"Label Tensor Shape: {label.shape}")

        optimizer.zero_grad()
        output = model(features)

        print(f"Model Output Shape: {output.shape}")  # Should match label shape

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        # Debugging: Print loss for every 5 batches
        if batch_idx % 5 == 0:
            print(f"[{CLIENT_NAME}] Training batch {batch_idx}, Loss: {loss.item()}")

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, label in dataloader:
            features, label = features.to(device), label.to(device)
            output = model(features)
            predicted = (output > 0.5).float()
            correct += (predicted == label).sum().item()
            total += label.size(0)
    return correct / total

# Flower Client for PyTorch
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLPModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.BCELoss()
        dataset = RecommendationDataset(df)
        self.dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.dataloader, self.optimizer, self.criterion, self.device)
        return self.get_parameters(config), len(df), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = evaluate(self.model, self.dataloader, self.device)

         # Ensure dataset has valid samples for evaluation
        dataset_length = max(1, len(df))  # Avoid division by zero

        return float(accuracy), dataset_length, {}

# Start Flower client
print(f"Starting {CLIENT_NAME} at {SERVER_ADDRESS}...")
fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=FlowerClient())
print(f"Starting {CLIENT_NAME} and connecting to Flower server at {SERVER_ADDRESS}...")

#code below is most successfull code for client
# import flwr as fl
# import numpy as np
# import os

# # Get server address from environment variables
# SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "172.20.0.2:8080")

# class FlowerClient(fl.client.NumPyClient):
#     def get_parameters(self, config):  # ✅ Accepts 'config'
#         print(f"[{os.getenv('CLIENT_NAME', 'Client')}] Sending model parameters to server...")
#         return np.array([])  # Replace with actual model weights if needed

#     def fit(self, parameters, config):
#         print(f"[{os.getenv('CLIENT_NAME', 'Client')}] Training model...")
#         return parameters, len(parameters), {}

#     def evaluate(self, parameters, config):
#         print(f"[{os.getenv('CLIENT_NAME', 'Client')}] Evaluating model...")
#         return 0.0, len(parameters), {}

# # Start Flower client with dynamic naming
# client_name = os.getenv("CLIENT_NAME", "fl_client")
# print(f"Starting {client_name}...")
# fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=FlowerClient())

# import flwr as fl
# import numpy as np

# class FlowerClient(fl.client.NumPyClient):
#     def get_parameters(self, config):  # ✅ Accepts 'config'
#         print("Sending model parameters to server...")
#         return np.array([])  # Replace with actual model weights if needed

#     def fit(self, parameters, config):
#         print("Training model...")
#         return parameters, len(parameters), {}

#     def evaluate(self, parameters, config):
#         print("Evaluating model...")
#         return 0.0, len(parameters), {}

# fl.client.start_numpy_client(server_address="172.20.0.2:8080", client=FlowerClient())
