import copy
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from beartype import beartype
from lightning import LightningDataModule
from pytorch_lightning.utilities.seed import isolate_rng
from torch import FloatTensor, Tensor
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", message=".*does not have many workers.*")


class SyntheticCopyDataset(Dataset):
    @beartype
    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        vocab_size: int,
        lookahead: int,
        datatype: str = "int"
    ):
        super().__init__()
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.lookahead = lookahead 
        self.datatype = datatype
        self.data, self.task_params = self.gen_data()

    @beartype
    def __len__(self) -> int:
        return self.n_samples

    @beartype
    def __getitem__(
        self, index
    ) -> Any:  # be careful of the return type, please read lightning doc for best-practices
        # Get the data for the given index
        x = self.data["x"][index]
        y = self.data["y"][index]
        return x, y

    @beartype
    @torch.inference_mode()
    def gen_data(
        self,
    ) -> Any:  # be careful on the return type

        if self.datatype == "int": 
            x = torch.randint(0, self.vocab_size, (self.n_samples, self.seq_len), dtype=torch.long)
            y = torch.full((self.n_samples, self.seq_len), -1, dtype=torch.long)  # Initialize with -1 for ignored positions
            y[:, self.lookahead:] = x[:, :-self.lookahead]

        elif self.datatype == "real":
            # Generate random sequences of real numbers
            x = torch.randn(self.n_samples, self.seq_len, self.vocab_size)

            # Create the output tensor with p padding tokens at the beginning
            y = torch.zeros_like(x)

            # Compute the shifted output
            y[:, self.lookahead:, :] = x[:, :-self.lookahead, :]

        # Make dicts for data and params 
        data_dict = {"x": x, "y": y}
        params_dict = {
            "seq_len": self.seq_len,
            "vocab_size": self.vocab_size,
            "lookahead": self.lookahead,
            "n_samples": self.n_samples,
            }

        return data_dict, params_dict


class SyntheticCopyDataModule(LightningDataModule):
    @beartype
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int,
        num_workers: int = 0,
        test_dataset: Optional[Dataset] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # setup

    @beartype
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=None,
        )

    @beartype
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=None,
        )

    @beartype
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=None,
        )

if __name__ == "__main__":
    # Example usage
    train_dataset = SyntheticCopyDataset(n_samples=1000, seq_len=10, vocab_size=5, lookahead=5)
    val_dataset = SyntheticCopyDataset(n_samples=200, seq_len=10, vocab_size=5, lookahead=5)
    test_dataset = SyntheticCopyDataset(n_samples=200, seq_len=10, vocab_size=5, lookahead=5)

    print(f"Train dataset size: {len(train_dataset)}")
    # convert train example to integers
    print(f"Train dataset example (int): {train_dataset[0][0]}")
    print(f"Train dataset example (int): {train_dataset[0][1]}")

    print(f"Test dataset size: {len(val_dataset)}")
    print(f"Test dataset example (int): {val_dataset[0][0]}")
    print(f"Test dataset example (int): {val_dataset[0][1]}")

    data_module = SyntheticCopyDataModule(train_dataset, val_dataset, batch_size=32, test_dataset=test_dataset)

    for batch in data_module.train_dataloader():
        print(batch[0].shape)
        break  # Just to show one batch

    for batch in data_module.val_dataloader():
        print(batch[0].shape)
        break  # Just to show one batch

    for batch in data_module.test_dataloader():
        print(batch[0].shape)
        break  # Just to show one batch

    real_train_dataset = SyntheticCopyDataset(
        n_samples=1000, seq_len=10, vocab_size=5, lookahead=5, datatype="real"
    )
    real_val_dataset = SyntheticCopyDataset(
        n_samples=200, seq_len=10, vocab_size=5, lookahead=5, datatype="real"
    )

    print("Train dataset size (real):", len(real_train_dataset))
    print("Test dataset size (real):", len(real_val_dataset))
    print("Train dataset example (real):", real_train_dataset[0][0])
    print("Train dataset example (real):", real_train_dataset[0][1])
    print("Test dataset example (real):", real_val_dataset[0][0])
    print("Test dataset example (real):", real_val_dataset[0][1])
    data_module_real = SyntheticCopyDataModule(
        real_train_dataset, real_val_dataset, batch_size=32
    )
    for batch in data_module_real.train_dataloader():
        print(batch[0].shape)
        break  # Just to show one batch