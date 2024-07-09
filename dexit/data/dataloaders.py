import logging
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class CIFARDataLoader:
    """
    A class for loading and preprocessing the CIFAR10 dataset.

    This class provides functionality to load both the CIFAR10 train and test datasets,
    apply necessary transformations, and create DataLoaders for efficient batch processing.
    It also allows limiting the number of samples in the test dataset.

    Attributes:
        batch_size (int): Number of samples per batch.
        root_dir (str): Root directory for storing the dataset.
        transform (transforms.Compose): Composition of image transformations.
        trainset (torchvision.datasets.CIFAR10): The CIFAR10 train dataset.
        testset (torchvision.datasets.CIFAR10): The CIFAR10 test dataset.
        trainloader (torch.utils.data.DataLoader): DataLoader for the train dataset.
        testloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        num_samples (int): Number of samples to use from the test dataset. If None, use all samples.
    """

    def __init__(self, batch_size: int = 4, root_dir: str = './shared/data', num_samples: int = None):
        """
        Initializes the CIFARDataLoader with the specified batch size, root directory, and number of samples.

        Args:
            batch_size (int): Number of samples per batch. Defaults to 4.
            root_dir (str): Root directory for storing the dataset. Defaults to './shared/data'.
            num_samples (int): Number of samples to use from the test dataset. If None, use all samples.
        """
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.num_samples = num_samples
        self.transform = self._create_transform()
        self.trainset = self._load_dataset(train=True)
        self.testset = self._load_dataset(train=False)
        self.trainloader = self._create_dataloader(self.trainset)
        self.testloader = self._create_dataloader(self.testset, limit_samples=True)

        logging.debug(f"CIFARDataLoader initialized with batch size: {self.batch_size}, num_samples: {self.num_samples}")

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

    def _load_dataset(self, train: bool) -> torchvision.datasets.CIFAR10:
        """
        Downloads and loads the CIFAR10 dataset.

        Args:
            train (bool): If True, loads the training dataset. If False, loads the test dataset.

        Returns:
            torchvision.datasets.CIFAR10: The loaded CIFAR10 dataset.
        """
        dataset = torchvision.datasets.CIFAR10(
            root=self.root_dir,
            train=train,
            download=True,
            transform=self.transform
        )
        logging.info(f"CIFAR10 {'train' if train else 'test'} dataset loaded from {self.root_dir}")
        return dataset

    def _create_dataloader(self, dataset: torchvision.datasets.CIFAR10, limit_samples: bool = False) -> DataLoader:
        """
        Creates a DataLoader for the given dataset.

        Args:
            dataset (torchvision.datasets.CIFAR10): The dataset to create a DataLoader for.
            limit_samples (bool): If True, limit the number of samples for the test dataset.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the dataset.
        """
        if limit_samples and self.num_samples is not None and dataset == self.testset:
            indices = torch.randperm(len(dataset))[:self.num_samples]
            dataset = Subset(dataset, indices)
            logging.info(f"Limited test dataset to {self.num_samples} samples")

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True if dataset == self.trainset else False,
            num_workers=2
        )
        logging.info(f"DataLoader created for {'train' if dataset == self.trainset else 'test'} dataset with batch size: {self.batch_size}")
        return dataloader

    def get_train_loader(self) -> DataLoader:
        """
        Returns the DataLoader for the train dataset.

        Returns:
            torch.utils.data.DataLoader: The DataLoader for the train dataset.
        """
        return self.trainloader

    def get_test_loader(self) -> DataLoader:
        """
        Returns the DataLoader for the test dataset.

        Returns:
            torch.utils.data.DataLoader: The DataLoader for the test dataset, potentially with limited samples.
        """
        return self.testloader

    def get_batch(self, train: bool = False) -> torch.utils.data.DataLoader:
        """
        Returns an iterator over the train or test dataset.

        Args:
            train (bool): If True, returns the train dataset iterator. If False, returns the test dataset iterator.

        Returns:
            torch.utils.data.DataLoader: An iterator over the specified dataset.
        """
        loader = self.trainloader if train else self.testloader
        logging.debug(f"Returning {'train' if train else 'test'} data iterator")
        return loader