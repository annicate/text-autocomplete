import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class NextTokenDataset(Dataset):
    def __init__(self, sequences):
        self.samples = []

        for seq in sequences:
            if len(seq) < 2:
                continue
            x = seq[:-1]
            y = seq[1:]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


def collate_fn(batch, pad_token_id):
    x_batch, y_batch = zip(*batch)

    x_batch = pad_sequence(
        x_batch,
        batch_first=True,
        padding_value=pad_token_id
    )

    y_batch = pad_sequence(
        y_batch,
        batch_first=True,
        padding_value=pad_token_id
    )

    return x_batch, y_batch