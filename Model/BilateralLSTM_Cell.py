import torch
import torch.nn as nn
import torch.nn.functional as F


class BilateralLSTMCell(nn.Module):
    """
    Translates Bilateral_lstm_class.py
    Computes LSTM gates using x, hidden_state_1 (self), and hidden_state_2 (coupled).
    """

    def __init__(self, input_dim, hidden_dim):
        super(BilateralLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim

        # We combine the inputs for efficiency: [x, h1, h2] -> Gates
        # The gates are Input (i), Forget (f), Cell (c_), Output (o)
        # So output dimension is 4 * hidden_dim
        self.linear = nn.Linear(input_dim + hidden_dim + hidden_dim, 4 * hidden_dim)

    def forward(self, x, h_self, c_self, h_coupled):
        # Concatenate inputs: x_t, h_{t-1}^{self}, h_{t-1}^{coupled}
        combined = torch.cat([x, h_self, h_coupled], dim=1)

        # Pass through linear layer to get all gate projections at once
        gates = self.linear(combined)

        # Split into individual gates
        i_gate, f_gate, o_gate, c_tilde = gates.chunk(4, dim=1)

        # Apply activations
        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        o = torch.sigmoid(o_gate)
        c_ = torch.tanh(c_tilde)

        # Update cell and hidden states
        c_next = f * c_self + i * c_
        h_next = o * torch.tanh(c_next)

        return h_next, c_next