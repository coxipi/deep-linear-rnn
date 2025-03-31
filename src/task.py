import random
from abc import ABC, abstractmethod
from typing import Any, Iterable, Literal

import torch
from beartype import beartype
from lightning import Callback, LightningModule
from torch import Tensor
from torch.nn import functional as F


def sequential_ce_loss(
    input: Tensor,
    target: Tensor,
    ignore_index: int = -1,
) -> Tensor:
    """
    Compute the negative log likelihood loss for a sequence of predictions.
    Args:
        input (Tensor): The predicted probabilities (shape: batch_size, seq_len, vocab_size).
        target (Tensor): The target indices (shape: batch_size, seq_len, vocab_size).
    Returns:
        Tensor: The computed loss.
    """
    # Reshape input and target to match the expected dimensions
    input = input.float()
    input = input.reshape(-1, input.size(-1))
    target = target.view(-1)

    # Compute the negative log likelihood loss
    loss = F.cross_entropy(input, target, ignore_index=ignore_index)
    return loss


class CopyTaskRegression(LightningModule):
    @beartype
    def __init__(
        self,
        model: Any,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters()  # ignore the instance of nn.Module that are already stored ignore=['my_module'])

    @beartype
    def forward(self, x: Any, batch_first=True) -> Any:
        # model takes seq_len first, but dataloader puts batch first...
        if batch_first:
            x = x.permute(1,0,2)
        seq_length, batch_size,  _ = x.size()
        h = self.model.init_hidden(batch_size)
        outputs = []
        
        for t in range(seq_length):
            out, h = self.model(x[t], h)
            outputs.append(out)
        
        outputs = torch.stack(outputs)
        # Revert back to match outputs with ground truth labels
        if batch_first:
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
        # Create mask for nonzero vectors (shape: seq_len, batch)
        mask = (y.abs().sum(dim=-1) != 0)
        # Compute loss only on nonzero vectors
        loss = self.loss_function(y[mask], preds[mask])
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


class CopyTaskTokenized(LightningModule):
    @beartype
    def __init__(
        self,
        model: Any,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters()  # ignore the instance of nn.Module that are already stored ignore=['my_module'])
        self.loss_function = sequential_ce_loss 

    @beartype
    def forward(self, x: Any, batch_first=True) -> Any:
        if batch_first:
            x = x.permute(1,0)

        seq_length, batch_size  = x.size()
        h = self.model.init_hidden(batch_size)
        outputs = []
        
        for t in range(seq_length):
            out, h = self.model(x[t], h)
            outputs.append(out)

        outputs = torch.stack(outputs)

        if batch_first:
            outputs = outputs.permute(1,0,2)

        return outputs 
    
    #def forward(self, x):
    #    x = x.permute(1,0)
    #    outs, _ = self.model(x)
    #    outs = outs.permute(1,0,2)
    #    print(outs.shape)
    #    # Revert back to match outputs with ground truth labels
    #    return outs

    @beartype
    def training_step(self, data, batch_idx) -> Tensor:
        x, y = data
        print("input shape:",x.shape)
        # Forward pass
        preds = self.forward(x)
        print("preds:", preds.shape)

        # Compute loss only on nonzero vectors
        loss = self.loss_function(preds, y)

        # Log loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    @beartype
    def validation_step(self, data, batch_idx) -> Tensor:
        x, y = data
        print("input shape:",x.shape)
        # Forward pass
        preds = self.forward(x)
        print("preds:", preds.shape)
        # Compute loss
        loss = self.loss_function(preds, y)
        # Log loss
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

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
    from model import DeepRNN, DeepRNNWithEmbedding, SimpleSeq2SeqRNN
    from dataset import SyntheticCopyDataset, SyntheticCopyDataModule
    seq_len = 10
    vocab_size = 2
    lookahead = 5
    embedding_size = 2
    hidden_size = 16 
    n_layers = 2


    # Example usage
    train_dataset = SyntheticCopyDataset(n_samples=1000, seq_len=seq_len, vocab_size=vocab_size, lookahead=lookahead, datatype="int")
    val_dataset = SyntheticCopyDataset(n_samples=200, seq_len=seq_len, vocab_size=vocab_size, lookahead=lookahead, datatype="int")
    data_module = SyntheticCopyDataModule(train_dataset, val_dataset, batch_size=32)

    #model = SimpleSeq2SeqRNN(vocab_size, embedding_size, hidden_size, vocab_size) 
    model = DeepRNNWithEmbedding(vocab_size, embedding_size, hidden_size, vocab_size, n_layers)
    task = CopyTaskTokenized(model=model, lr=1e-3)
    # Initialize the trainer
    from lightning import Trainer
    trainer = Trainer(max_epochs=100, accelerator="cpu")
    # Train the model
    trainer.fit(task, data_module)
    # Validate the model
    trainer.validate(task, data_module)



