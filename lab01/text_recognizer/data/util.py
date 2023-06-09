from typing import Any, Callable, Dict, Sequence, Tuple, Union
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
        super().__init__()
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


