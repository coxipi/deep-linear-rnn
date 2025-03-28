from abc import ABC, abstractmethod
from typing import Any

import torch
from beartype import beartype
from torch import Tensor, nn
from torch.nn import functional as F
from ssm import S4Block as S4

class abstract_model(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Perform forward pass of the model."""
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F


# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d



class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=128,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4(d_model, dropout=dropout, transposed=True)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x


class DeepRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, activation='relu', output_type='real', readout_activation='linear'):
        super(DeepRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_type = output_type
        
        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'tanh':
            self.activation_fn = torch.tanh
        elif activation == 'identity':
            self.activation_fn = lambda x: x
        else:
            raise ValueError("activation must be 'relu', 'tanh', or 'identity'")
        
        if readout_activation == 'relu':
            self.readout_activation_fn = F.relu
        elif readout_activation == 'tanh':
            self.readout_activation_fn = torch.tanh
        elif readout_activation == 'linear':
            self.readout_activation_fn = lambda x: x
        else:
            raise ValueError("readout_activation must be 'relu', 'tanh', or 'linear'")
        
        self.rnn_layers = nn.ModuleList([
            nn.RNNCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        h_next = []
        for i, rnn in enumerate(self.rnn_layers):
            h_i = rnn(x if i == 0 else h_next[i-1], h[i])
            h_i = self.activation_fn(h_i)
            h_next.append(h_i)
        out = self.fc(h_next[-1])
        out = self.readout_activation_fn(out)
        
        if self.output_type == 'token':
            out = F.log_softmax(out, dim=-1)
        
        return out, torch.stack(h_next)
    
    def init_hidden(self, batch_size):
        return [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]


def main():
    input_size = 5
    hidden_size = 10
    output_size = 5
    num_layers = 3
    seq_length = 10 
    batch_size = 32
    
    test_cases = [
        ('identity', 'linear'),
        ('relu', 'relu'),
        ('identity', 'relu'),
        ('relu', 'linear')
    ]
    
    x = torch.randn(seq_length, batch_size, input_size)
    
    for activation, readout_activation in test_cases:
        model = DeepRNN(input_size, hidden_size, output_size, num_layers, activation, 'real', readout_activation)
        h = model.init_hidden(batch_size)
        outputs = []
        
        for t in range(seq_length):
            out, h = model(x[t], h)
            outputs.append(out)
        
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2)
        print(f"Model with activation={activation}, readout_activation={readout_activation} -> Output shape: {outputs.shape}")

            # Test case for token input with a two-letter alphabet (one-hot encoding)
    token_input_size = 2  # Two-letter alphabet
    token_output_size = 2
    
    token_model = DeepRNN(token_input_size, hidden_size, token_output_size, num_layers, 'relu', 'token', 'linear')
    token_h = token_model.init_hidden(batch_size)
    
    token_x = torch.randint(0, 2, (seq_length, batch_size))  # Random binary sequence
    token_x = F.one_hot(token_x, num_classes=token_input_size).float()  # Convert to OHE
    
    token_outputs = []
    for t in range(seq_length):
        out, token_h = token_model(token_x[t], token_h)
        token_outputs.append(out)
    
    token_outputs = torch.stack(token_outputs)
    print(f"Token model (OHE input) -> Output shape: {token_outputs.shape}")

    # Test S4 model
    x = torch.randn(batch_size, seq_length, input_size)
    s4_model = S4Model(d_input=input_size, d_output=output_size, d_model=hidden_size, n_layers=num_layers)
    token_outputs = []
    print("throw the full input to s4 layer")
    print(s4_model(x).shape)


if __name__ == "__main__":
    main()