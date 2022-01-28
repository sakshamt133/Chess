from torch.utils.data import random_split, DataLoader
from dataset import Chess
import utils


data = Chess(utils.path, utils.transforms)

train_len = int(0.9 * len(data))
test_len = len(data) - train_len

train, test = random_split(data, [train_len, test_len])

train_data = DataLoader(
    train, utils.batch_size, shuffle=True
)

test_data = DataLoader(
    test, utils.batch_size
)