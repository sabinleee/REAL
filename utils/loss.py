import torch.nn as nn

# define a loss function
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, sample, pred):
        return self.mse(sample['depth'], pred)
