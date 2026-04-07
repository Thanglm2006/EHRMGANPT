import torch
import torch.nn as nn


class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=3):
        super(VAE_Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        # Defensive clamps to prevent mathematical explosions
        mu = torch.clamp(mu, min=-20.0, max=20.0)
        logvar = torch.clamp(logvar, min=-20.0, max=20.0)

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
        out, _ = self.lstm(z)
        logits = self.fc_out(out)
        reconstruction = torch.sigmoid(logits)
        return reconstruction, logits


class SequenceDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super(SequenceDiscriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.fc(out).squeeze(-1)
        return logits, out


class BilateralLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BilateralLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(input_dim + hidden_dim + hidden_dim, 4 * hidden_dim)

    def forward(self, x, h_self, c_self, h_coupled):
        combined = torch.cat([x, h_self, h_coupled], dim=1)
        gates = self.linear(combined)
        i_gate, f_gate, o_gate, c_tilde = gates.chunk(4, dim=1)

        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        o = torch.sigmoid(o_gate)
        c_ = torch.tanh(c_tilde)

        c_next = f * c_self + i * c_
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class BilateralGenerator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, latent_dim, num_layers=3):
        super(BilateralGenerator, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.cells = nn.ModuleList([
            BilateralLSTMCell(
                input_dim=noise_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim
            ) for i in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, noise_seq, h_coupled_states):
        batch_size, time_steps, _ = noise_seq.size()
        h_states = [torch.zeros(batch_size, self.hidden_dim, device=noise_seq.device) for _ in range(self.num_layers)]
        c_states = [torch.zeros(batch_size, self.hidden_dim, device=noise_seq.device) for _ in range(self.num_layers)]
        outputs = []

        for t in range(time_steps):
            x_t = noise_seq[:, t, :]
            for i in range(self.num_layers):
                h_cpl = h_coupled_states[i]
                h_states[i], c_states[i] = self.cells[i](x_t, h_states[i], c_states[i], h_cpl)
                x_t = h_states[i]

            out_t = torch.sigmoid(self.fc_out(x_t))
            outputs.append(out_t.unsqueeze(1))

        return torch.cat(outputs, dim=1), h_states