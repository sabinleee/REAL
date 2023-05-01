# import libraries
from torch.utils.data import Dataset, DataLoader

# define a dataset
class Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

# define a dataloader
def dataloader(data, batch_size=1, shuffle=False, num_workers=0):
    dataset = Dataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader