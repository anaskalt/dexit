import torchvision
from torchvision import transforms
import torch

# Define the transformation to be applied to each image in the dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
])

# Batch size for the data loader
batch_size = 4

# Download and load the CIFAR10 test dataset
testset = torchvision.datasets.CIFAR10(
    root='./shared/data',  # Root directory where the dataset will be stored
    train=False,  # Specify that we want the test set
    download=True,  # Download the dataset if not already downloaded
    transform=transform  # Apply the defined transformations
)

# Create a data loader for the test set
testloader = torch.utils.data.DataLoader(
    testset,  # The dataset to load
    batch_size=batch_size,  # Number of samples per batch
    shuffle=False,  # Do not shuffle the dataset
    num_workers=2  # Number of subprocesses to use for data loading
)
