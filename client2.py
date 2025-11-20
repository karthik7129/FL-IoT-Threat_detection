import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flwr.client import NumPyClient, start_client
from collections import OrderedDict
from model import NeuralNetwork

def load_and_preprocess_data():
    """
    Load and preprocess data from device_2_reduced.csv following notebook approach
    """
    # Read the reduced dataset
    df = pd.read_csv("ReducedData/device_4_reduced.csv")
    print(f"Dataset shape: {df.shape}")

    # Filter for labels 0,1,2,3,5,6,7,8,9 (as in notebook)
    df_filtered = df[df['label'].isin([0, 1, 2, 3, 5, 6, 7, 8, 9])].copy()
    print(f"Filtered dataset shape: {df_filtered.shape}")

    # Separate features and labels
    feature_columns = [col for col in df_filtered.columns if col != 'label']
    X = df_filtered[feature_columns].values
    y = df_filtered['label'].values

    # Standardize features (as in notebook)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def load_data():
    """
    Load and preprocess data, split into train/test sets
    """
    # Load and preprocess data
    X, y = load_and_preprocess_data()

    # Train-test split (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, y_train, X_test, y_test

def train(model, train_loader, optimizer, criterion):
    """
    Train the model for one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    return train_loss, train_accuracy

def test(model, test_loader, criterion):
    """
    Evaluate the model
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = total_loss / len(test_loader)
    test_accuracy = correct / total
    return test_loss, test_accuracy

# Load data
X_train, y_train, X_test, y_test = load_data()

# Get input size and number of classes
input_size = X_train.shape[1]
num_classes = 10

print(f"Input size: {input_size}, Number of classes: {num_classes}")

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 264
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
net = NeuralNetwork(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Train for 5 epochs (as in notebook)
        for epoch in range(5):
            train_loss, train_accuracy = train(net, train_loader, optimizer, criterion)

        print(f"Client 2 - Training completed. Accuracy: {train_accuracy:.4f}")

        return self.get_parameters(config={}), len(train_loader.dataset), {
            "loss": train_loss,
            "accuracy": train_accuracy,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, test_loader, criterion)
        print(f"Client 2 - Evaluation accuracy: {accuracy:.4f}")
        return loss, len(test_loader.dataset), {"accuracy": accuracy}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Federated Learning Client 2')
    parser.add_argument(
        '--server',
        type=str,
        default=os.getenv('FL_SERVER_ADDRESS', 'localhost:8080'),
        help='Server address in format host:port (default: localhost:8080)'
    )
    args = parser.parse_args()
    
    print(f"Client 2 connecting to server: {args.server}")
    start_client(
        server_address=args.server,
        client=FlowerClient(),
    )
