from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np


def load_mnist(batch_size):
    # classes = list(range(10))
    classes = [0, 1, 3]
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = MNIST('./data', transform=img_transform)
    dataset = Subset(dataset, np.where(np.isin(dataset.targets, classes))[0])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Modality label. sub0 contains feature indices for modality 1. These
    # are the first pixel, third pixel, 5th, etc, so every other pixel.
    # The remaning pixel indices are stored in sub1 for modality 2. This
    # one is shuffled to make sure feature n of modality 1 is not correlated
    # with feature n of modality 2.
    sub0 = np.array(range(0, 784, 2))
    sub1 = sub0 + 1
    np.random.seed(0)
    np.random.shuffle(sub1)

    return dataloader, sub0, sub1
