import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalSelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim // 8)
        self.key = nn.Linear(hidden_dim, hidden_dim // 8)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x shape: (batch_size, time_steps, hidden_dim)
        batch_size, time_steps, hidden_dim = x.size()
        
        proj_query = self.query(x).view(batch_size, -1, time_steps).permute(0, 2, 1) # B x T x C
        proj_key = self.key(x).view(batch_size, -1, time_steps) # B x C x T
        
        energy = torch.bmm(proj_query, proj_key) # B x T x T
        attention = torch.softmax(energy, dim=-1)
        
        proj_value = self.value(x).view(batch_size, -1, time_steps) # B x C x T
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B x C x T
        out = out.view(batch_size, time_steps, hidden_dim)
        
        out = self.gamma * out + x
        return out


class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=3):
        super(VAE_Encoder, self).__init__()
        # Avoid dropout error when num_layers = 1
        dropout_rate = 0.2 if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x_t, hidden_state):
        # x_t shape: [batch_size, 1, input_dim]
        out, new_hidden = self.lstm(x_t, hidden_state)
        out_squeeze = out.squeeze(1)

        mu = self.fc_mu(out_squeeze)
        logvar = self.fc_logvar(out_squeeze)

        # Defensive clamps to prevent numerical instability
        mu = torch.clamp(mu, min=-20.0, max=20.0)
        logvar = torch.clamp(logvar, min=-20.0, max=20.0)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar, new_hidden


class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=3):
        super(VAE_Decoder, self).__init__()
        dropout_rate = 0.2 if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z_t, hidden_state):
        # z_t shape: [batch_size, 1, latent_dim]
        out, new_hidden = self.lstm(z_t, hidden_state)
        logits = self.fc_out(out.squeeze(1))
        reconstruction = torch.sigmoid(logits)
        return reconstruction, logits, new_hidden


class AutoregressiveVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, enc_layers, dec_layers, time_steps):
        super(AutoregressiveVAE, self).__init__()
        self.time_steps = time_steps
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # cVAE Architecture: Input Encoder = Original Input (x_t) + Error (x_hat) -> dim = input_dim * 2
        self.encoder = VAE_Encoder(input_dim * 2, hidden_dim, latent_dim, enc_layers)
        self.decoder = VAE_Decoder(latent_dim, hidden_dim, input_dim, dec_layers)

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        # Initialize hidden states
        enc_hidden = (torch.zeros(self.encoder.lstm.num_layers, batch_size, self.hidden_dim, device=device),
                      torch.zeros(self.encoder.lstm.num_layers, batch_size, self.hidden_dim, device=device))
        dec_hidden = (torch.zeros(self.decoder.lstm.num_layers, batch_size, self.hidden_dim, device=device),
                      torch.zeros(self.decoder.lstm.num_layers, batch_size, self.hidden_dim, device=device))

        c_prev = torch.zeros(batch_size, self.input_dim, device=device)

        rec_list, logits_list, mu_list, logvar_list, z_list = [], [], [], [], []

        for t in range(self.time_steps):
            x_t = x[:, t, :]
            c_sigmoid = torch.sigmoid(c_prev)

            # Calculate residual error following VRNN logic
            x_hat = x_t - c_sigmoid

            # Pass through encoder for each timestep
            enc_in = torch.cat([x_t, x_hat], dim=1).unsqueeze(1)
            z_t, mu_t, logvar_t, enc_hidden = self.encoder(enc_in, enc_hidden)

            # Decode to get output for the next iteration
            z_in = z_t.unsqueeze(1)
            rec_t, logits_t, dec_hidden = self.decoder(z_in, dec_hidden)

            c_prev = logits_t  # Save logits (pre-sigmoid output)

            rec_list.append(rec_t.unsqueeze(1))
            logits_list.append(logits_t.unsqueeze(1))
            mu_list.append(mu_t.unsqueeze(1))
            logvar_list.append(logvar_t.unsqueeze(1))
            z_list.append(z_t.unsqueeze(1))

        return torch.cat(rec_list, dim=1), torch.cat(logits_list, dim=1), \
            torch.cat(mu_list, dim=1), torch.cat(logvar_list, dim=1), torch.cat(z_list, dim=1)

    def reconstruct_decoder(self, z_seq):
        """Function used for GAN Generator when sequence noise z is already provided"""
        batch_size = z_seq.size(0)
        device = z_seq.device
        dec_hidden = (torch.zeros(self.decoder.lstm.num_layers, batch_size, self.hidden_dim, device=device),
                      torch.zeros(self.decoder.lstm.num_layers, batch_size, self.hidden_dim, device=device))

        rec_list, logits_list = [], []
        for t in range(self.time_steps):
            z_t = z_seq[:, t, :].unsqueeze(1)
            rec_t, logits_t, dec_hidden = self.decoder(z_t, dec_hidden)
            rec_list.append(rec_t.unsqueeze(1))
            logits_list.append(logits_t.unsqueeze(1))

        return torch.cat(rec_list, dim=1), torch.cat(logits_list, dim=1)


class SequenceDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_steps, num_layers=3):
        super(SequenceDiscriminator, self).__init__()
        dropout_rate = 0.2 if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        # Add Self Attention for advanced global pooling
        self.attn = TemporalSelfAttention(hidden_dim)
        # Use spectral normalization to enforce Lipschitz continuity for WGAN-GP
        self.fc = spectral_norm(nn.Linear(hidden_dim * time_steps, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.attn(out)
        out_flat = torch.flatten(out, start_dim=1)
        logits = self.fc(out_flat).squeeze(-1)  # Output an unbounded critic score for WGAN
        return logits, out


class BilateralLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BilateralLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.ln = nn.Linear(input_dim + hidden_dim + hidden_dim, 4 * hidden_dim, bias=False)

    def forward(self, x, h_self, c_self, h_coupled):
        combined = torch.cat([x, h_self, h_coupled], dim=1)
        gates = self.ln(combined)
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

        self.cl = nn.ModuleList([
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
                h_states[i], c_states[i] = self.cl[i](x_t, h_states[i], c_states[i], h_cpl)
                x_t = h_states[i]

            out_t = torch.sigmoid(self.fc_out(x_t))
            outputs.append(out_t.unsqueeze(1))

        return torch.cat(outputs, dim=1), h_states


class JointGenerator(nn.Module):
    def __init__(self, c_noise_dim, d_noise_dim, hidden_dim, c_latent_dim, d_latent_dim, num_layers=3):
        super(JointGenerator, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.c_gen = BilateralGenerator(c_noise_dim, hidden_dim, c_latent_dim, num_layers)
        self.d_gen = BilateralGenerator(d_noise_dim, hidden_dim, d_latent_dim, num_layers)

        # Advanced Attention to resolve generator mode-collapse on patterns
        self.c_attn = TemporalSelfAttention(hidden_dim)
        self.d_attn = TemporalSelfAttention(hidden_dim)

    def forward(self, noise_c, noise_d):
        """
        Implements the step-by-step bilateral coupling required by EHR-M-GAN,
        enhanced with sequence-level Self-Attention for continuous distributions.
        """
        batch_size, time_steps, _ = noise_c.size()
        device = noise_c.device

        # Initialize hidden and cell states for both streams
        c_h = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.num_layers)]
        c_c = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.num_layers)]
        d_h = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.num_layers)]
        d_c = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.num_layers)]

        c_features_list = []
        d_features_list = []

        for t in range(time_steps):
            # 1. Capture current noise inputs
            noise_c_t = noise_c[:, t, :]
            noise_d_t = noise_d[:, t, :]

            # 2. Coupled inputs come from the OTHER stream's PREVIOUS hidden state
            c_h_coupled = d_h
            d_h_coupled = c_h

            # 3. Step C Generator Layers
            c_x = noise_c_t
            for i in range(self.num_layers):
                c_h[i], c_c[i] = self.c_gen.cl[i](c_x, c_h[i], c_c[i], c_h_coupled[i])
                c_x = c_h[i]
            
            c_features_list.append(c_x.unsqueeze(1))

            # 4. Step D Generator Layers
            d_x = noise_d_t
            for i in range(self.num_layers):
                d_h[i], d_c[i] = self.d_gen.cl[i](d_x, d_h[i], d_c[i], d_h_coupled[i])
                d_x = d_h[i]
            
            d_features_list.append(d_x.unsqueeze(1))

        # Apply Global Attention across the sequence to fix discrete pattern loss
        c_temporal = torch.cat(c_features_list, dim=1)
        d_temporal = torch.cat(d_features_list, dim=1)

        c_attn_out = self.c_attn(c_temporal)
        d_attn_out = self.d_attn(d_temporal)

        # Removed restrictive sigmoid constraint on generator output.
        # WGAN natively handles unconstrained generated latents (matches VAE true N(mu, sigma) mapping).
        fake_z_c = self.c_gen.fc_out(c_attn_out)
        fake_z_d = self.d_gen.fc_out(d_attn_out)

        return fake_z_c, fake_z_d