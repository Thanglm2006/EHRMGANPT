from Model.BilateralLSTM_Cell import BilateralLSTMCell
import torch
import torch.nn as nn
import torch.nn.functional as F

class BilateralGenerator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, latent_dim, num_layers=3):
        super(BilateralGenerator, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Create multiple layers of Bilateral Cells
        self.cells = nn.ModuleList([
            BilateralLSTMCell(
                input_dim=noise_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim
            ) for i in range(num_layers)
        ])

        # Final linear projection to the shared latent space dimension
        self.fc_out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, noise_seq, h_coupled_states):
        """
        noise_seq: (batch_size, time_steps, noise_dim)
        h_coupled_states: Hidden states from the *other* generator
        """
        batch_size, time_steps, _ = noise_seq.size()

        # Initialize hidden and cell states for all layers
        h_states = [torch.zeros(batch_size, self.hidden_dim, device=noise_seq.device) for _ in range(self.num_layers)]
        c_states = [torch.zeros(batch_size, self.hidden_dim, device=noise_seq.device) for _ in range(self.num_layers)]

        outputs = []

        for t in range(time_steps):
            x_t = noise_seq[:, t, :]

            # Pass through each layer
            for i in range(self.num_layers):
                h_cpl = h_coupled_states[i]  # The coupled hidden state from the parallel generator
                h_states[i], c_states[i] = self.cells[i](x_t, h_states[i], c_states[i], h_cpl)
                x_t = h_states[i]  # Output of current layer is input to the next

            # Final projection for this time step
            out_t = torch.sigmoid(self.fc_out(x_t))
            outputs.append(out_t.unsqueeze(1))

        return torch.cat(outputs, dim=1), h_states