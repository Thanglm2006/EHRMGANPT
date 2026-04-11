import torch
import torch.nn as nn


class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=3):
        super(VAE_Encoder, self).__init__()
        # Tránh lỗi dropout khi num_layers = 1
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

        # Defensive clamps
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

        # Kiến trúc cVAE: Input Encoder = Input Gốc (x_t) + Error (x_hat) -> dim = input_dim * 2
        self.encoder = VAE_Encoder(input_dim * 2, hidden_dim, latent_dim, enc_layers)
        self.decoder = VAE_Decoder(latent_dim, hidden_dim, input_dim, dec_layers)

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        # Khởi tạo state
        enc_hidden = (torch.zeros(self.encoder.lstm.num_layers, batch_size, self.hidden_dim, device=device),
                      torch.zeros(self.encoder.lstm.num_layers, batch_size, self.hidden_dim, device=device))
        dec_hidden = (torch.zeros(self.decoder.lstm.num_layers, batch_size, self.hidden_dim, device=device),
                      torch.zeros(self.decoder.lstm.num_layers, batch_size, self.hidden_dim, device=device))

        c_prev = torch.zeros(batch_size, self.input_dim, device=device)

        rec_list, logits_list, mu_list, logvar_list, z_list = [], [], [], [], []

        for t in range(self.time_steps):
            x_t = x[:, t, :]
            c_sigmoid = torch.sigmoid(c_prev)

            # Tính phần dư theo đúng VRNN của bạn
            x_hat = x_t - c_sigmoid

            # Đưa qua encoder từng timestep
            enc_in = torch.cat([x_t, x_hat], dim=1).unsqueeze(1)
            z_t, mu_t, logvar_t, enc_hidden = self.encoder(enc_in, enc_hidden)

            # Giải mã bằng decoder để lấy output cho vòng lặp sau
            z_in = z_t.unsqueeze(1)
            rec_t, logits_t, dec_hidden = self.decoder(z_in, dec_hidden)

            c_prev = logits_t  # Lưu lại logits (đầu ra trước khi qua sigmoid)

            rec_list.append(rec_t.unsqueeze(1))
            logits_list.append(logits_t.unsqueeze(1))
            mu_list.append(mu_t.unsqueeze(1))
            logvar_list.append(logvar_t.unsqueeze(1))
            z_list.append(z_t.unsqueeze(1))

        return torch.cat(rec_list, dim=1), torch.cat(logits_list, dim=1), \
            torch.cat(mu_list, dim=1), torch.cat(logvar_list, dim=1), torch.cat(z_list, dim=1)

    def reconstruct_decoder(self, z_seq):
        """Hàm dùng cho GAN Generator khi đã có trước chuỗi nhiễu z"""
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
        # Giống hệt tf.layers.flatten(outputs)
        self.fc = nn.Linear(hidden_dim * time_steps, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out_flat = torch.flatten(out, start_dim=1)
        logits = self.fc(out_flat).squeeze(-1)  # Output 1 điểm duy nhất
        return logits, out


class BilateralLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BilateralLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(input_dim + hidden_dim + hidden_dim, 4 * hidden_dim, bias=False)

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