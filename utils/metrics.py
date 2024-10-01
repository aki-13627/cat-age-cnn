import torch.nn as nn
import torch.optim as optim

def get_criterion():
    return nn.CrossEntropyLoss()

def get_optimizer(model, learning_rate=0.001):
    return optim.Adam(model.fc.parameters(), lr=learning_rate)
