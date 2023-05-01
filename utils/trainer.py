import torch.nn as nn

# define a trainer
class Trainer(nn.Module):
    def __init__(self, model, loss, optimizer, device):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
    
    def forward(self, sample):
        sample = sample.to(self.device)
        pred = self.model(sample)
        loss = self.loss(sample, pred)
        self.optimizer(loss)
        return loss.item()
    