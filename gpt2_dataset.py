import torch
from torch.utils.data import Dataset

class GPT2Dataset(Dataset):

    def __init__(self, tokens_file):
        with open(tokens_file, "r") as file:
            tokens = file.readlines()
        self.tokens = [int(token.strip()) for token in tokens]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return torch.tensor(self.tokens[index])
