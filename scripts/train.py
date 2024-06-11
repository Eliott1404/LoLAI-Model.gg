import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import mlflow
import mlflow.pytorch
from data_prep import load_data
from model import NeuralNetwork
import yaml

with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)


# Set the tracking URI to your local filesystem (default)
mlflow.set_tracking_uri("http://localhost:5000")

# Set the experiment name
mlflow.set_experiment("League of Legends Win Prediction")

# Set MLflow experiment
mlflow.set_tracking_uri(config['mlflow']['uri'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

# Load data
X_train, X_test, y_train, y_test = load_data(config['data']['file_path'])

# Convert to PyTorch tensors
X_train, y_train = torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.long)
X_test, y_test = torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

# Model, loss function, optimizer
model = NeuralNetwork(X_train.shape[1], config['model']['hidden_size'], config['model']['output_size'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

# Train the model
with mlflow.start_run():
    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        mlflow.log_metric("loss", running_loss / len(train_loader), step=epoch)

    mlflow.pytorch.log_model(model, "model")

    # Save the model
    torch.save(model.state_dict(), config['model']['save_path'])

print("Training completed")
