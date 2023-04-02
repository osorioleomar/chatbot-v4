import torch
from torch.utils.data import Dataset

class GPT2Dataset(Dataset):
    def __init__(self, tokens_file, block_size=128):
        with open(tokens_file, "r") as file:
            tokens = [int(token.strip()) for token in file.readlines()]

        self.block_size = block_size
        self.data = torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = (idx + 1) * self.block_size
        return self.data[start:end]
