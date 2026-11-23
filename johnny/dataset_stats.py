import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def calculate_mean_std(dataset_path) -> tuple[list[float], list[float]]:
    """
    Find the dataset's mean and standard deviation for each channel

    :param dataset_path: the path to the dataset.
    :type dataset_path: string
    :return: two lists, the first is the mean and the second is the std (3 values in each, for each channel)
    :rtype: tuple[list[float], list[float]]
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    num_pixels = 0

    for images, _ in loader:
        batch_size, _, height, width = images.shape
        num_pixels += batch_size * height * width
        
        channels_sum += images.sum(dim=[0, 2, 3])
        channels_squared_sum += (images ** 2).sum(dim=[0, 2, 3])

    mean = channels_sum / num_pixels
    std = torch.sqrt((channels_squared_sum / num_pixels) - (mean ** 2))

    return mean, std