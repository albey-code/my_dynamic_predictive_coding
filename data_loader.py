import os.path as op
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
#import torchvision.transforms as transforms

class VideoDataset(Dataset):

    def __init__(self, data_path):
        """
        Args:
            data_path: (string) path to the birdman testing data file
        """
        super(VideoDataset, self).__init__()
        if not op.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found at {data_path}")
        d = np.load(data_path)  # Load birdman data #check whether the shape really is 7200, 10, 256
        assert d.shape == (360000, 10, 256), f"Unexpected data shape: {d.shape}"
        self.data = torch.tensor(d).float()

    def __len__(self):
        return self.data.shape[0]  # Number of sequences (7200)

    def __getitem__(self, idx):
        return self.data[idx]  # Return a single sequence of shape (10, 256)

def fetch_dataloader(data_dir, params):
    """
    Fetches the DataLoader object for the birdman_test dataset.

    Args:
        data_dir: (string) directory containing the birdman dataset
        params: (Params) hyperparameters

    Returns:
        test_loader: (DataLoader) DataLoader object for the birdman_test dataset
    """
    data_path = op.join(data_dir, "birdman_test.npy")  # Update file name to match your dataset
    dl = DataLoader(
        VideoDataset(data_path),
        batch_size=params.batch_size,
        shuffle=False,  # No shuffling for testing
        num_workers=params.num_workers,
        pin_memory=params.cuda
    )
    return dl
