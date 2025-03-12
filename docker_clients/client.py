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
CLIENT_NAME = os.getenv("CLIENT_NAME", "fl_client_1")

if CLIENT_NAME == "fl_client_1":
    data_path = "/app/data/Partitioned_Client_1_Pet_Supplies.csv"
elif CLIENT_NAME == "fl_client_2":
    data_path = "/app/data/Partitioned_Client_2_Baby_Skin_and_Grooming.csv"
else:
    raise ValueError("Invalid client name!")

# Load dataset
df = pd.read_csv(data_path)
print(f"[{CLIENT_NAME}] Loaded dataset with {len(df)} records.")

# Recommendation dataset class
class RecommendationDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.users = torch.randint(0, 1000, (len(dataframe),))  
        self.items = torch.randint(0, 500, (len(dataframe),))   
        self.ratings = torch.randint(0, 2, (len(dataframe),))  

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

# Neural Collaborative Filtering (NCF) model
class NCFModel(nn.Module):
    def __init__(self, num_users=1000, num_items=500, embedding_dim=16):
        super(NCFModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, user, item):
        user_emb = self.user_embedding(user)  # (batch_size, embedding_dim)
        item_emb = self.item_embedding(item)  # (batch_size, embedding_dim)

        x = torch.cat([user_emb, item_emb], dim=-1)  # (batch_size, embedding_dim * 2)

        # Debugging Print to Verify Shape Before Linear Layers
        print(f"User Embedding Shape: {user_emb.shape}")
        print(f"Item Embedding Shape: {item_emb.shape}")
        print(f"Concatenated Input Shape: {x.shape}")

        # Ensure x is flattened properly before passing into the Linear layers
        x = x.view(x.shape[0], -1)  # Flatten (batch_size, embedding_dim * 2)

        return self.fc_layers(x).squeeze()  # Ensure the output shape is correct

# Training function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    for batch_idx, (user, item, label) in enumerate(dataloader):  # ✅ Added `enumerate()` to define batch_idx
        user, item, label = user.to(device), item.to(device), label.to(device, dtype=torch.float)

        # Debugging Prints
        print(f"User Tensor Shape: {user.shape}")
        print(f"Item Tensor Shape: {item.shape}")
        print(f"Label Tensor Shape: {label.shape}")

        optimizer.zero_grad()
        output = model(user, item)

        print(f"Model Output Shape: {output.shape}")  # Should match label shape

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        # Debugging: Print loss for every 5 batches
        if batch_idx % 5 == 0:
            print(f"[{CLIENT_NAME}] Training batch {batch_idx}, Loss: {loss.item()}")
# def train(model, dataloader, optimizer, criterion, device):
#     model.train()
#     for user, item, label in dataloader:
#         user, item, label = user.to(device), item.to(device), label.to(device, dtype=torch.float)
#         optimizer.zero_grad()
#         output = model(user, item)
#         loss = criterion(output, label)
#         loss.backward()
#         optimizer.step()

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for user, item, label in dataloader:
            user, item, label = user.to(device), item.to(device), label.to(device, dtype=torch.float)
            output = model(user, item)
            predicted = (output > 0.5).float()
            correct += (predicted == label).sum().item()
            total += label.size(0)
    return correct / total

# Flower Client for PyTorch
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NCFModel().to(self.device)
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

