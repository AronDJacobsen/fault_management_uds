from pathlib import Path

#import typer
from loguru import logger
from tqdm import tqdm
from torch.utils.data import Dataset
import torch


from fault_management_uds.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


class TimeSeriesDataset(Dataset):
    def __init__(self, data, start_indices, sequence_length):
        self.data = data
        self.start_indices = start_indices
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        return torch.tensor(self.data[self.start_indices[idx]:self.start_indices[idx] + self.sequence_length], dtype=torch.float32)

