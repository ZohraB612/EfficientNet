from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
import random

class Tensor(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data


def my_collate(batch):
    # B x Set(S) x Strokes(s)

    # Ensure each item in the batch is a tensor
    Set = pad_sequence(batch, batch_first=True)

    # Strokes
    R = []

    samp = 256
    for idx, val in enumerate(batch):
        # Ensure the batch item is a tensor
        if not isinstance(batch[idx], torch.Tensor):
            raise TypeError(f"Expected batch item to be a torch.Tensor, but got {type(batch[idx])} instead.")
        
        # Debug output to inspect the shape and type of the item
        # print(f"Processing batch item {idx} of type {type(batch[idx])} with shape {batch[idx].shape}")

        # Sampling strokes if the item is a sequence (tensor)
        if len(batch[idx]) <= samp:
            Randomly_sampled_strokes = random.choices(batch[idx], k=samp)
        else:
            Randomly_sampled_strokes = random.sample(list(batch[idx]), k=samp)  # Explicitly convert to list

        R.append(torch.stack(Randomly_sampled_strokes, dim=0))

    Strokes = torch.stack(R, dim=0)

    return Set, Strokes
