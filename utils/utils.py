import os
import albumentations
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import pandas as pd

# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #

class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        self.size = size
        self.images =  self._get_image_paths(path)
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    # def _get_image_paths(self, path):
    #     image_paths = []
    #     for root, _, files in os.walk(path):
    #         for file in files:
    #             if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
    #                 image_paths.append(os.path.join(root, file))
    #     return image_paths
    


    def _get_image_paths(self, train_file):
        # Initialize an empty list to store paths
        image_paths = []
        
        # Read the train.txt file
        with open(train_file, 'r') as f:
            for line in f:
                # Each line contains "path label", split by space
                path, label = line.strip().split()
                image_paths.append(path)
        
        return image_paths

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example

#-------------------------------------------------------------------------------------
class EEGImageDataset(Dataset):
    def __init__(self, eeg_data, labels, images, size = None):
        self.size = size
        self.eeg_data = eeg_data
        self.labels = labels
        self.images = images

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return len(self.labels)

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, idx):
        eeg = torch.tensor(self.eeg_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        try:
            img = self.preprocess_image(self.images.loc[idx, "image_paths"])
        except:
            return self.__getitem__(idx%self.__len__())
        return eeg, label, img


def load_transformer_data(args, train_data_path='processed_data/v2/train_data.npz', train_csv_path = "processed_data/v2/train_image_data.csv"):
    # Load EEG data
    train_data = np.load(train_data_path, mmap_mode="r",allow_pickle=True)
    # test_data = np.load(test_data_path,  mmap_mode="r", allow_pickle=True)
    
    # Load image paths and labels from CSV
    train_image_data = pd.read_csv(train_csv_path)
    # test_image_data = pd.read_csv(test_csv_path)

    # Create datasets
    train_dataset = EEGImageDataset(
        eeg_data=train_data['eeg'],
        labels=train_data['labels'],
        images = train_image_data,
        size=args.image_size
    )

    # test_dataset = EEGImageDataset(
    #     eeg_data=test_data['eeg'],
    #     labels=test_data['labels']
    # )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers = 4)
    # test_loader = DataLoader(test_dataset, batch_size=train_batch_size, shuffle=False, drop_last=True, num_workers = 4)
    return train_loader


def load_data(args):
    train_data = ImagePaths(args.dataset_path, size=args.image_size)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)
    return train_loader


# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()
