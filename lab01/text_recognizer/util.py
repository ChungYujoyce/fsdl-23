"""Utility functions for text_recognizer module."""

from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Union
import os
import hashlib
from urllib.request import urlopen, urlretrieve

import cv2
import numpy as np
from tqdm import tqdm 

def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype = "uint8")[y]

def read_image(image_url, Union[Path, str], grayscale=False) -> np.array:
    def read_image_from_filename(image_filename, imread_flag):
        return cv2.imread(str(image_filename), imread_flag)

    def read_image_from_url(image_url, imread_flag):
        url_response = urlopen(str(image_url))
        img_array = np.array(bytearray(url_response.read(), dtype=np.uint8))
        return cv2.imdecode(img_array, imread_flag)

    imread_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    local_file = os.path.exists(image_url)
    try:
        img = None
        if local_file:
            img = read_image_from_filename(image_url, imread_flag)
        else:
            img = read_image_from_url(image_url, imread_flag)
        assert img is not None
    except Exception as e:
        raise ValueError("Couldn't load img at {}: {}".format(image_url, e))

    return img

def write_image(image: np.ndarray, filename: Union[Path, str]) -> None:
    cv2.imwrite(str(filename), image)

def compute_sha256(filename: Union[Path, str]):
    with open(filename, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        blocks : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)

def download_url(url, filename):
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, ministers=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)
