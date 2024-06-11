import torch
import torch.nn as nn
import torch.nn.functional as F

class WideAndDeepModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WideAndDeepModel, self).__init__()
        
        # Wide part
        self.wide = nn.Linear(input_size, output_size)
        
        # Deep part
        self.deep_fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.6)
        
        self.deep_fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.7)
        
        self.deep_fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Wide part
        wide_output = self.wide(x)
        
        # Deep part
        x = F.relu(self.bn1(self.deep_fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.deep_fc2(x)))
        x = self.dropout2(x)
        
        deep_output = self.deep_fc3(x)
        
        # Combine wide and deep parts
        output = wide_output + deep_output
        return output