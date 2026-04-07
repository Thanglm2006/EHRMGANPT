import torch
import torch.nn as nn
class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=3):
        super(VAE_Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        # Shared sampling layers mapping hidden state -> latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, (h_n, c_n) = self.lstm(x)

        # Get mu and logvar for every time step
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar


class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=3):
        super(VAE_Decoder, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # z shape: (batch, seq_len, latent_dim)
        out, _ = self.lstm(z)
        # Output reconstruction (sigmoid because data is normalized/binary)
        reconstruction = torch.sigmoid(self.fc_out(out))
        return reconstruction