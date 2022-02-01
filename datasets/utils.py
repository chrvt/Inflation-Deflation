import os
import numpy as np
import torch
import requests
import pandas as pd
from matplotlib import pyplot as plt

from torch.utils.data import Dataset


def _cartesian_to_spherical(inputs_):
    # Calculate hyperspherical coordinates one by one
    dim = inputs_.shape[1]-1
    inputs = torch.from_numpy(inputs_)
    phis = []
    for i in range(dim):
        r_ = 1 #torch.sum(inputs[:, i : dim + 1] ** 2, dim=1) ** 0.5
        phi_ = torch.acos(inputs[:, i] / r_)

        # The cartesian -> spherical transformation is not unique when inputs_i to inputs_n are all zero
        # In that case we can choose to set the coordinate to 0
        # This choice maybe avoids derivatives evaluating to NaNs? Noo... :/
        # phi_ = torch.where(r_ < 0.04, torch.zeros_like(phi_), phi_)

        # Actually, we have to be more aggressive to save the gradients from becoming NaNs!
        # When inputs_(i+1) to inputs_n are all zero, the argument to the arccos is very small,
        # either below zero (when inputs_i is negative) or above (when inputs_i is positive).
        # In this case we can fix this angle to be zero or pi.
        # But it also doesn't suffice...
        # phi_ = torch.where(
        #     torch.sum(inputs[:, i+1 : dim + 1] ** 2, dim=1) < 0.000001,
        #     torch.where(
        #         inputs[:,i] < 0.,
        #         np.pi * torch.ones_like(phi_),
        #         torch.zeros_like(phi_)
        #     ),
        #     phi_
        # )
        # logger.debug(torch.sum(torch.sum(inputs[:, i+1 : dim + 1] ** 2, dim=1) < 0.000001).item())

        phi_ = phi_.view((-1, 1))
        phis.append(phi_)

    # Special case for last component, see https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
    # import pdb
    # pdb.set_trace()
    phis[-1] = torch.where(inputs[:, dim] < 0.0, 2.0 * np.pi - phis[-1][:, 0], phis[-1][:, 0]).view((-1, 1))
    
    # Rescale                                                                      - self.azimuthal_offset
    # phis = [-1.0 + 2.0 * phi / np.pi if i < dim - 1 else -1.0 + torch.remainder(phi , 2.0 * np.pi) / np.pi for i, phi in enumerate(phis)]

    # Radial coordinate
    # r = torch.sum(inputs[:, : dim + 1] ** 2, dim=1) ** 0.5
    # dr = r #- self.r0
    # dr = dr.view((-1, 1))

    # Combine
    # others = inputs[:, dim + 1 :]
    outputs = torch.cat(phis, dim=1) #+ [dr, others]
    # outputs = torch.cat(phis + [dr, others], dim=1)

    return outputs.numpy()


def download_file(url, dest):
    CHUNK_SIZE = 8192

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # Filter out keep-alive new chunks.
                f.write(chunk)


def download_file_from_google_drive(id, dest):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, dest)


class NumpyDataset(Dataset):
    """ Dataset for numpy arrays with explicit memmap support """

    def __init__(self, *arrays, **kwargs):

        self.dtype = kwargs.get("dtype", torch.float)
        self.memmap = []
        self.data = []
        self.n = None

        memmap_threshold = kwargs.get("memmap_threshold", None)

        for array in arrays:
            if isinstance(array, str):
                array = self._load_array_from_file(array, memmap_threshold)

            if self.n is None:
                self.n = array.shape[0]
            assert array.shape[0] == self.n

            if isinstance(array, np.memmap):
                self.memmap.append(True)
                self.data.append(array)
            else:
                self.memmap.append(False)
                tensor = torch.from_numpy(array).to(self.dtype)
                self.data.append(tensor)

    def __getitem__(self, index):
        items = []
        for memmap, array in zip(self.memmap, self.data):
            if memmap:
                tensor = np.array(array[index])
                items.append(torch.from_numpy(tensor).to(self.dtype))
            else:
                items.append(array[index])
        return tuple(items)

    def __len__(self):
        return self.n

    @staticmethod
    def _load_array_from_file(filename, memmap_threshold_gb=None):
        filesize_gb = os.stat(filename).st_size / 1.0 * 1024 ** 3
        if memmap_threshold_gb is None or filesize_gb <= memmap_threshold_gb:
            data = np.load(filename)
        else:
            data = np.load(filename, mmap_mode="c")

        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        return data

class NumpyValidationSet(Dataset):
    def __init__(self, x, transform=None):
        self.transform = transform
        self.x = torch.from_numpy(x)

    def __getitem__(self, index):
        x = self.x[index, ...]

        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return self.x.shape[0]

class LabelledImageDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.transform = transform
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        x = self.x[index, ...]
        y = self.y[index, ...]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.x.shape[0]
    
import logging
logger = logging.getLogger(__name__)

class UnlabelledImageDataset(Dataset):
    def __init__(self, array, transform=None):
        self.transform = transform
        self.data = torch.from_numpy(array)

    def __getitem__(self, index):
        img = self.data[index, ...] 
        #logger.info('img with index %s',index)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor([0.0])

    def __len__(self):
        return self.data.shape[0]


class CSVLabelledImageDataset(Dataset):
    """ Based on https://pytorch.org/tutorials/beginner/data_loading_tutorial.html """

    def __init__(self, csv_file, root_dir, label_key, filename_key, image_transform=None, label_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.label_key = label_key
        self.filename_key = filename_key

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_filename = os.path.join(self.root_dir, self.df[self.filename_key].iloc[idx])
        image = torch.from_numpy(np.transpose(plt.imread(img_filename), [2, 0, 1]))
        label = torch.tensor([self.df[self.label_key].iloc[idx]], dtype=np.float)

        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            label = self.label_transform(label)

        return image, label


class Preprocess:
    def __init__(self, num_bits):
        self.num_bits = num_bits
        self.num_bins = 2 ** self.num_bits

    def __call__(self, img):
        if img.dtype == torch.uint8:
            img = img.float()  # Already in [0,255]
        else:
            img = img * 255.0  # [0,1] -> [0,255]

        if self.num_bits != 8:
            img = torch.floor(img / 2 ** (8 - self.num_bits))  # [0, 255] -> [0, num_bins - 1]

        # Uniform dequantization.
        img = img + torch.rand_like(img)

        return img

    def inverse(self, inputs):
        # Discretize the pixel values.
        inputs = torch.floor(inputs)
        # Convert to a float in [0, 1].
        inputs = inputs * (256 / self.num_bins) / 255
        inputs = torch.clamp(inputs, 0, 1)
        return inputs


class RandomHorizontalFlipTensor(object):
    """Random horizontal flip of a CHW image tensor."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        assert img.dim() == 3
        if np.random.rand() < self.p:
            return img.flip(2)  # Flip the width dimension, assuming img shape is CHW.
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)
