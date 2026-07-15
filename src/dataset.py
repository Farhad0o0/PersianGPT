import random

import torch
import numpy as np

from torch.utils.data import IterableDataset


def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        seed = self.base_seed
    else:
        seed = self.base_seed + worker_info.id  # هر worker seed متفاوت

    rng = np.random.default_rng(seed)
    while True:
        idx = rng.integers(0, len(self.tokens) - self.seq_len)
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        yield torch.from_numpy(chunk[:-1].astype(np.int64)), torch.from_numpy(chunk[1:].astype(np.int64))


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