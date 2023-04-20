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

def convert_strings_to_labels(strings: Sequence[str], mapping: Dict[str, int], length: int) -> torch.Tensor:
    """
    Convert sequence of N strings to a (N, length) ndarray, with each string wrapped with <S> and <E> tokens,
    and padded with the <P> token.
    """
    labels = torch.ones((len(strings), length), dtype=torch.long) * mapping["<P>"]
    for i, string in enumerate(strings):
        tokens = list(string)
        tokens = ["<S>", *tokens, "<E>"]
        for ii, token in enumerate(tokens):
            labels[i, ii] = mapping[token]
    return labels


def split_dataset(base_dataset: BaseDataset, fraction: float, seed: int) -> Tuple[BaseDataset, BaseDataset]:
    """
    Split input base_dataset into 2 base datasets, the first of size fraction * size of the base_dataset and the
    other of size (1 - fraction) * size of the base_dataset.
    """
    split_a_size = int(fraction * len(base_dataset))
    split_b_size = len(base_dataset) - split_a_size
    return torch.utils.data.random_split(  # type: ignore
        base_dataset, [split_a_size, split_b_size], generator=torch.Generator().manual_seed(seed)
    )


