from pathlib import Path
from typing import Collection, Dict, Optional, Tuple, Union
import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

from text_recognizer import util
from text_recognizer.data.util import BaseDataset

def load_and_print_info(data_module_class: type) -> None:
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    print(dataset)


def _download_raw_dataset(metadata: Dict, dl_dirname: Path) -> Path: # func for internal use
    dl_dirname.mkdir(parents=True, exist_ok=True)
    filename = dl_dirname / metadata["filename"]
    if filename.exists():
        return filename
    print(f"Downloading raw dataset from {metadata['url']} to {filename}...")
    util.download_url(metadata["url"], filename)
    
    sha256 = util.compute_sha256(filename)
    if sha256 != metadata["sha256"]:
        raise ValueError("Downloaded data file SHA-256 does not match that listed in metadata document.")
    return filename

BATCH_SIZE = 128
NUM_WORKERS = 0

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)
        
        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        # Make sure to set the variables below in subclasses
        self.dims: Tuple[int, ...]
        self.output_dims: Tuple[int, ...]
        self.mapping: Collection
        self.data_train: Union[BaseDataset, ConcatDataset]
        self.data_val: Union[BaseDataset, ConcatDataset]
        self.data_test: Union[BaseDataset, ConcatDataset]


    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[3] / "data"

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=BATCH_SIZE)
        parser.add_argument(
            "--num_workers", type=int, default=NUM_WORKERS)
        parser.add_argument(
            "--subsample_fraction",
            type=float,
            default=None,
            help="If given, is used as the fraction of data to expose.")
        return parser

    def config(self):
        return {"input_dims": self.dims, "output_dims": self.output_dims, "mapping": self.mapping}

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.on_gpu)

    def val_dataloader(self):
        return DataLoader(self.data_val, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.on_gpu)

    def test_dataloader(self):
        return DataLoader(self.data_test, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.on_gpu)