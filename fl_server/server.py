import flwr as fl
import torch
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define the PyTorch model
class FLModel(torch.nn.Module):
    def __init__(self):
        super(FLModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Define a Flower server strategy
class FLServer(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        print(f"Round {rnd}: Aggregating {len(results)} updates")
        return super().aggregate_fit(rnd, results, failures)

# Start the FL server
def start_fl_server():
    print("Starting Flower FL Server...")
    fl.server.start_server(server_address="0.0.0.0:8080", strategy=FLServer())

@app.route('/receive_model', methods=['POST'])
def receive_model():
    model_weights = request.json['weights']
    print(f"Received model update from client: {np.array(model_weights).shape}")
    return jsonify({"message": "Model update received"})

if __name__ == '__main__':
    start_fl_server()
    app.run(host='0.0.0.0', port=5000, debug=True)
