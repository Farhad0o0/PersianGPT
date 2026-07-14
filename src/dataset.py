import random

import torch
import numpy as np

from torch.utils.data import IterableDataset



class PersianDataset(IterableDataset):

    def __init__(self, tokens, seq_len):

        self.tokens = tokens
        self.seq_len = seq_len

    def __iter__(self):

        max_start = len(self.tokens) - self.seq_len - 1

        while True:

            start = random.randint(0, max_start)

            x = self.tokens[start:start+self.seq_len]

            y = self.tokens[start+1:start+self.seq_len+1]

            yield (
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long)
            )