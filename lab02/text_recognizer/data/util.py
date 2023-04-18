from typing import Any, Callable, Sequence, Tuple, Union, Dict
import torch

sequence_or_tensor = Union[Sequence, torch.Tensor]

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, 
        data: sequence_or_tensor,
        targets: sequence_or_tensor,
        transform: Callable = None,
        target_transform: Callable = None) -> None:

        if len(data) != len(targets):
            raise ValueError("Data and targets must be of equal length")
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        datum, target = self.data[index], self.targets[index]

        if self.transform is not None:
            datum = self.transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target

def split_dataset(base_dataset: BaseDataset, fraction: float, seed: int) -> Tuple[BaseDataset, BaseDataset]:
    split_a_size = int(fraction * len(base_dataset))
    split_b_size = len(base_dataset) - split_a_size
    return torch.utils.data.random_split(  # type: ignore
        base_dataset, [split_a_size, split_b_size], generator=torch.Generator().manual_seed(seed)
    )



