import torch
import torch.nn as nn
    
    
# define a optimizer
class Optimizer(nn.Module):
    def __init__(self, model, lr=0.001):
        super(Optimizer, self).__init__()
        self.model = model
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def forward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        