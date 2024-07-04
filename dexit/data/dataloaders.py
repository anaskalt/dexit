import logging
from typing import Tuple, Iterator
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class CIFARDataLoader:
    """
    A class for loading and preprocessing the CIFAR10 dataset.

    This class provides functionality to load the CIFAR10 test dataset,
    apply necessary transformations, and create a DataLoader for efficient
    batch processing.

    Attributes:
        batch_size (int): Number of samples per batch.
        root_dir (str): Root directory for storing the dataset.
        transform (transforms.Compose): Composition of image transformations.
        testset (torchvision.datasets.CIFAR10): The CIFAR10 test dataset.
        testloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    """

    def __init__(self, batch_size: int = 4, root_dir: str = './shared/data'):
        """
        Initializes the CIFARDataLoader with the specified batch size and root directory.

        Args:
            batch_size (int): Number of samples per batch. Defaults to 4.
            root_dir (str): Root directory for storing the dataset. Defaults to './shared/data'.
        """
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.transform = self._create_transform()
        self.testset = self._load_dataset()
        self.testloader = self._create_dataloader()

        logging.debug(f"CIFARDataLoader initialized with batch size: {self.batch_size}")

    def _create_transform(self) -> transforms.Compose:
        """
        Creates a composition of image transformations to be applied to the dataset.

        Returns:
            transforms.Compose: A composition of image transformations.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        logging.debug("Image transformation pipeline created")
        return transform

    def _load_dataset(self) -> torchvision.datasets.CIFAR10:
        """
        Downloads and loads the CIFAR10 test dataset.

        Returns:
            torchvision.datasets.CIFAR10: The loaded CIFAR10 test dataset.
        """
        testset = torchvision.datasets.CIFAR10(
            root=self.root_dir,
            train=False,
            download=True,
            transform=self.transform
        )
        logging.info(f"CIFAR10 test dataset loaded from {self.root_dir}")
        return testset

    def _create_dataloader(self) -> DataLoader:
        """
        Creates a DataLoader for the test dataset.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the test dataset.
        """
        testloader = DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        logging.info(f"DataLoader created with batch size: {self.batch_size}")
        return testloader

    def get_loader(self) -> DataLoader:
        """
        Returns the DataLoader for the test dataset.

        Returns:
            torch.utils.data.DataLoader: The DataLoader for the test dataset.
        """
        return self.testloader

    def get_batch(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns an iterator over the test dataset.

        Yields:
            Tuple[torch.Tensor, torch.Tensor]: A batch of input tensors and their corresponding labels.
        """
        for inputs, labels in self.testloader:
            logging.debug(f"Yielding batch: inputs shape {inputs.shape}, labels shape {labels.shape}")
            yield inputs, labels