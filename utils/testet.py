import torch.nn as nn

# define a tester
class Tester(nn.Module):
    def __init__(self, model, device):
        super(Tester, self).__init__()
        self.model = model
        self.device = device
    
    def forward(self, sample):
        sample = sample.to(self.device)
        pred = self.model(sample)
        return pred
    