import random
from abc import ABC, abstractmethod
from typing import Any, Iterable, Literal

import torch
from beartype import beartype
from lightning import Callback, LightningModule
from torch import Tensor
from torch.nn import functional as F


class CopyTask(LightningModule):
    @beartype
    def __init__(
        self,
        model: Any,
        lr: float = 1e-4,
        loss: str = "bce",
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters()  # ignore the instance of nn.Module that are already stored ignore=['my_module'])
        self.loss = loss

    @beartype
    def forward(self, x: Any) -> Any:
        # model takes seq_len first, but dataloader puts batch first...
        x = x.permute(1,0,2)
        seq_length, batch_size,  _ = x.size()
        h = self.model.init_hidden(batch_size)
        outputs = []
        
        for t in range(seq_length):
            out, h = self.model(x[t], h)
            outputs.append(out)
        
        outputs = torch.stack(outputs)
        # Revert back to match outputs with ground truth labels
        outputs = outputs.permute(1,0,2)
        return outputs 


    @beartype
    def training_step(self, data, batch_idx) -> Tensor:
        x, y = data
        # Forward pass
        preds = self.forward(x)
        # Create mask for nonzero vectors (shape: seq_len, batch)
        mask = (y.abs().sum(dim=-1) != 0)
        # Compute loss only on nonzero vectors
        loss = self.loss_function(y[mask], preds[mask])

        # Log loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    @beartype
    def validation_step(self, data, batch_idx) -> Tensor:
        x, y = data
        # Forward pass
        preds = self.forward(x)
        # Compute loss
        loss = self.loss_function(y, preds)
        # Log loss
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss


    @abstractmethod
    def loss_function(self, target: Any, preds: Any) -> Tensor:
        """Loss function to be used in the training loop."""
        if "mse":
            loss = F.mse_loss(preds, target)
        elif "bce":
            loss = F.binary_cross_entropy(preds, target)
        return loss

    def configure_optimizers(self) -> None:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

class MyCustomCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_start(self, trainer, pl_module) -> None:  # pl_module is the LightningModule
        # setattr(pl_module, "my_param", new_param)
        pass

if __name__ == "__main__":
    from model import DeepRNN
    from dataset import SyntheticCopyDataset, SyntheticCopyDataModule

    # Example usage
    train_dataset = SyntheticCopyDataset(n_samples=1000, seq_len=10, vocab_size=5, lookahead=5, datatype="real")
    val_dataset = SyntheticCopyDataset(n_samples=200, seq_len=10, vocab_size=5, lookahead=5, datatype="real")
    model = DeepRNN(input_size=5, hidden_size=256, output_size=5, num_layers=4, activation='relu', output_type='real', readout_activation='linear')
    data_module = SyntheticCopyDataModule(train_dataset, val_dataset, batch_size=32)

    task = CopyTask(model=model, lr=1e-3, loss='mse')
    # Initialize the trainer
    from lightning import Trainer
    trainer = Trainer(max_epochs=500, accelerator="cpu")
    # Train the model
    trainer.fit(task, data_module)
    # Validate the model
    trainer.validate(task, data_module)



