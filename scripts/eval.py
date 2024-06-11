import torch
from sklearn.metrics import classification_report
from data_prep import load_data
from model import NeuralNetwork
import yaml

with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load data
X_train, X_test, y_train, y_test = load_data(config['data']['file_path'])

# Convert to PyTorch tensors
X_test, y_test = torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.long)

# Load model
model = NeuralNetwork(X_test.shape[1], config['model']['hidden_size'], config['model']['output_size'])
model.load_state_dict(torch.load(config['model']['save_path']))
model.eval()

# Evaluate the model
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    report = classification_report(y_test, predicted, output_dict=True)
    print(report)
