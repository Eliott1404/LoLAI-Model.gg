import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score
from data_prep import load_data
from model import NeuralNetwork
from utils import setup_logging
import yaml
import os
import pandas as pd

logger = setup_logging()

# Load configuration
with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

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
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

# Model, loss function, optimizer
model = NeuralNetwork(X_train.shape[1], config['model']['hidden_size'], config['model']['output_size'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler']['step_size'], gamma=config['scheduler']['gamma'])

# Checkpoint directory
checkpoint_dir = config['training']['checkpoint_dir']
os.makedirs(checkpoint_dir, exist_ok=True)

# Early stopping parameters
best_val_loss = float('inf')
patience = config['training']['patience']
counter = 0

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

        avg_train_loss = running_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(test_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("accuracy", accuracy, step=epoch)
        logger.info(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        # Step the scheduler
        scheduler.step()

        # Checkpoint and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            logger.info('Model checkpoint saved.')
        else:
            counter += 1
            if counter >= patience:
                logger.info('Early stopping triggered.')
                break

    # Log the best model
    mlflow.pytorch.log_model(model, "model")
    torch.save(model.state_dict(), config['model']['save_path'])

    # Evaluate the model using mlflow.evaluate
    eval_data = pd.DataFrame(X_test.numpy())
    eval_data["label"] = y_test.numpy()
    model_uri = mlflow.get_artifact_uri("model")

    result = mlflow.evaluate(
        model_uri,
        eval_data,
        targets="label",
        model_type="classifier",
        evaluators=["default"]
    )

logger.info("Training completed")
