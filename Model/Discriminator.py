import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    Translates C_GAN_NET.build_Discriminator and D_GAN_NET.build_Discriminator
    """

    def __init__(self, input_dim, hidden_dim, seq_len, num_layers=3, dropout=0.2):
        super(Discriminator, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        # Flatten the entire sequence of hidden states for the final classification
        self.fc = nn.Linear(hidden_dim * seq_len, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        features, _ = self.lstm(x)  # features shape: (batch, seq_len, hidden_dim)

        # Flatten features for the dense layer
        features_flat = features.reshape(features.size(0), -1)

        # Output un-activated logits (we will use BCEWithLogitsLoss later for numerical stability)
        logits = self.fc(features_flat)

        return logits, features