"Datasets related features and implementations of custom PyTorch datasets"

import os
import tarfile
import PIL

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url
from tqdm.notebook import tqdm

from src.utils.global_constants import DATA_DIR


class CUB200Dataset(Dataset):
    """Custom PyTorch Dataset class for CUB200-2011

    Description: https://huggingface.co/datasets/cassiekang/cub200_dataset/blob/main/README.md
    """

    url = (
        "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
    )
    md5 = "97eceeb196236b17998738112f37df78"
    filename = "CUB_200_2011.tgz"

    def __init__(
        self,
        download_dir: str = DATA_DIR,
        train: bool = True,
        download: bool = True,
        transform=None,
    ):
        # Download archive with data if necessary
        if download and not os.path.exists(os.path.join(download_dir, self.filename)):
            download_url(self.url, download_dir, self.filename, self.md5)
        # Unpack downloaded archive if there is one
        if not os.path.exists(
            os.path.join(download_dir, "CUB_200_2011")
        ) and os.path.exists(os.path.join(download_dir, self.filename)):
            with tarfile.open(os.path.join(download_dir, self.filename), "r:gz") as tar:
                tar.extractall(path=download_dir)
        # Raise error if dataset directory wasn't created
        if not os.path.exists(os.path.join(download_dir, "CUB_200_2011")):
            raise RuntimeError("Dataset isn't downloaded. Repeat with 'download=True'")
        self.dataset_dir = os.path.join(download_dir, "CUB_200_2011")
        self.transform = transform
        self.image_ids = self._load_train_or_test_ids(train)
        self.image_paths = self._load_image_paths()
        self.class_labels = self._load_class_labels()

    def _load_train_or_test_ids(self, train: bool) -> set:
        """Get a list of image IDs that belong to the specified train or test split"""
        ids = []
        with open(
            os.path.join(self.dataset_dir, "train_test_split.txt"),
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                img_id, img_is_train = (int(s) for s in line.strip().split())
                if img_is_train == train:
                    ids.append(img_id)
        return ids

    def _load_image_paths(self) -> dict:
        """Get a dict with image IDs and corresponding paths"""
        paths = {}
        with open(
            os.path.join(self.dataset_dir, "images.txt"), "r", encoding="utf-8"
        ) as f:
            for line in f:
                img_id, img_path = line.strip().split()
                img_id = int(img_id)
                if img_id in self.image_ids:
                    paths[img_id] = os.path.join(self.dataset_dir, "images", img_path)
        return paths

    def _load_class_labels(self) -> dict:
        """Get a dict with image IDs and corresponding class labels"""
        labels = {}
        with open(
            os.path.join(self.dataset_dir, "image_class_labels.txt"),
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                img_id, img_class_label = (int(s) for s in line.strip().split())
                if img_id in self.image_ids:
                    labels[img_id] = img_class_label
        return labels

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        img_id = self.image_ids[idx]
        img_path = self.image_paths[img_id]
        class_label = self.class_labels[img_id]
        img = PIL.Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, class_label


class EmbeddingDataset(Dataset):
    """Custom PyTorch Dataset class for loading embeddings and labels from disk.

    Designed for datasets where embeddings and their corresponding labels are
    stored as PyTorch tensors on a disk.
    """

    def __init__(self, filename: str):
        self.data = torch.load(os.path.join(DATA_DIR, filename))

    def __len__(self) -> int:
        return len(self.data["labels"])

    def __getitem__(self, idx: int) -> dict:
        embedding = self.data["embeddings"][idx]
        label = self.data["labels"][idx]
        return embedding, label


def extract_embeddings(
    dataloader: DataLoader,
    model: nn.Module,
    output_file: str,
    device: torch.device = torch.device("cpu"),
    standardize: bool = True,
) -> None:
    """Acquire embeddings from the model, standardize and save them to output_file
    alongside with the corresponding labels for future use with EmbeddingDataset.
    """

    output_file = os.path.join(DATA_DIR, output_file)
    # Prevent repeated extraction
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. The operation is aborted")
        return
    model = model.to(device)
    model.eval()
    outputs = []
    targets = []
    with torch.no_grad():
        for batch_inputs, batch_targets in tqdm(
            dataloader, desc=f"Extracting embeddings to {output_file}"
        ):
            batch_inputs = batch_inputs.to(device)
            batch_outputs = model(batch_inputs)
            outputs.append(batch_outputs.cpu())
            targets.append(batch_targets)
    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    # Make class labels start from 0
    targets = targets - targets.min()
    # Standardize the embedding's features (mean=0, std=1 in every column)
    if standardize:
        per_feature_mean = outputs.mean(dim=0, keepdim=True)
        per_feature_std = outputs.std(dim=0, keepdim=True)
        outputs = (outputs - per_feature_mean) / per_feature_std
    torch.save(
        {"embeddings": outputs, "labels": targets},
        output_file,
    )
