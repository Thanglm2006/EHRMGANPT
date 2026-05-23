# Hướng Dẫn Cải Tiến M3GAN: Khắc Phục Mode Collapse Dữ Liệu Liên Tục

Tài liệu này tổng hợp phân tích sâu sắc về nguyên nhân gây ra hiện tượng **Mode Collapse (Sụp đổ chế độ)** đối với đặc trưng liên tục (continuous features) trong kiến trúc `neo_m3gan` và cung cấp hướng dẫn lập trình chi tiết để bạn tự tay nâng cấp hệ thống.

---

## 1. Phân Tích Nguyên Nhân Cốt Lõi

Sau khi đánh giá mã nguồn hiện tại, có 4 nhân tố chính gây ra Mode Collapse trên phân phối liên tục:

1.  **Lỗi Logic Generator Mismatch (Lỗi Nặng Nhất):** Lúc huấn luyện, Generator tạo ra mã ẩn latent liên kết không bị ràng buộc (không qua Sigmoid) và có lớp Attention chuỗi thời gian (`c_attn`). Nhưng lúc chạy Fast Metric Evaluation và Checkpoint Visualization, code lại gọi riêng lẻ `c_gen` và `d_gen` (luồng này hoàn toàn **không có Attention** và **bị ép qua Sigmoid** làm sai lệch phân phối toán học của Latent, dẫn đến việc chọn nhầm checkpoint kém và Early Stopping sai).
2.  **Thiếu Phản Hồi Batch Diversity:** Discriminator đánh giá từng mẫu riêng lẻ, khiến Generator chỉ sinh ra duy nhất một mẫu "an toàn nhất" để đánh lừa Critic thay vì sinh đa dạng (định nghĩa gốc của Mode Collapse).
3.  **Discriminator Chưa Đạt Cận Tối Ưu:** `d_rounds` mặc định bằng `2` là quá nhỏ để Critic hội tụ trong WGAN-GP. Ngoài ra, việc dùng cùng learning rate (`1e-4`) làm mất đi nguyên lý TTUR (Two-Timescale Update Rule).
4.  **Hiện Tượng KL Vanishing (Posterior Collapse):** Do VAE sử dụng LSTM tự hồi quy làm Decoder, VAE pretraining rất dễ triệt tiêu KL Loss về 0 và bỏ qua mã ẩn latent, khiến không gian latent của GAN bị rỗng thông tin.

---

## 2. Các Bước Cải Tiến Chi Tiết & Mã Nguồn Sẵn Có

Dưới đây là các phần mã nguồn được thiết kế tối ưu, an toàn và đồng bộ hoàn toàn với các phần còn lại trong workspace của bạn.

### Bước 1: Thêm Minibatch Standard Deviation vào Discriminator (`neo_m3gan/networks.py`)

Thêm tính năng tính độ lệch chuẩn của cả lô (Batch) vào đầu vào của LSTM trong Discriminator để nhận diện các lô dữ liệu nghèo nàn (thiếu đa dạng).

**Vị trí sửa:** Tìm lớp `SequenceDiscriminator` trong `neo_m3gan/networks.py` (khoảng dòng 141-157) và cập nhật thành:

```python
class SequenceDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_steps, num_layers=3):
        super(SequenceDiscriminator, self).__init__()
        dropout_rate = 0.2 if num_layers > 1 else 0.0
        # Tăng thêm 1 chiều đầu vào (input_dim + 1) để chứa kênh thống kê Minibatch StdDev
        self.lstm = nn.LSTM(input_dim + 1, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        # Giữ nguyên Self Attention cho advanced global pooling
        self.attn = TemporalSelfAttention(hidden_dim)
        # Giữ nguyên spectral normalization để đảm bảo tính Lipschitz continuity cho WGAN-GP
        self.fc = spectral_norm(nn.Linear(hidden_dim * time_steps, 1))

    def forward(self, x):
        # x shape: [batch_size, time_steps, input_dim]
        batch_size, time_steps, channels = x.size()
        
        # 1. Tính độ lệch chuẩn theo chiều Batch
        # std shape: [1, time_steps, channels]
        std = torch.std(x, dim=0, keepdim=True) + 1e-8
        
        # 2. Lấy trung bình cộng trên tất cả đặc trưng để thu được 1 đặc trưng thống kê chung
        # std_mean shape: [1, time_steps, 1]
        std_mean = std.mean(dim=-1, keepdim=True)
        
        # 3. Repeat đặc trưng này tương ứng với kích thước Batch
        # std_mean shape: [batch_size, time_steps, 1]
        std_mean = std_mean.repeat(batch_size, 1, 1)
        
        # 4. Ghép nối kênh độ lệch chuẩn vào chuỗi đầu vào gốc
        # x_concat shape: [batch_size, time_steps, input_dim + 1]
        x_concat = torch.cat([x, std_mean], dim=-1)

        out, _ = self.lstm(x_concat)
        out = self.attn(out)
        out_flat = torch.flatten(out, start_dim=1)
        logits = self.fc(out_flat).squeeze(-1)  # Đầu ra critic score không ràng buộc cho WGAN
        return logits, out
```

---

### Bước 2: Khắc Phục Lỗi Mismatch Generator & Thêm KL Annealing (`neo_m3gan/trainer.py`)

#### A. Đồng bộ hóa luồng sinh dữ liệu trong Fast Metric Evaluation:
Tìm hàm `train_m3gan` trong `neo_m3gan/trainer.py`, đi tới block `Fast Metric Evaluation` (khoảng dòng 353-370) và cập nhật thành:

```python
            c_gen_eval, d_gen_eval = [], []
            with torch.no_grad():
                for _ in range(eval_batches):
                    noise_c = torch.randn(config['batch_size'], time_steps, noise_dim, device=device)
                    noise_d = torch.randn(config['batch_size'], time_steps, noise_dim, device=device)

                    # ROOT FIX: Huấn luyện dùng joint_gen thì đánh giá cũng PHẢI dùng joint_gen!
                    # Không dùng c_gen/d_gen riêng lẻ vì chúng bị thiếu Attention và bị ép qua Sigmoid.
                    fake_z_c, fake_z_d = joint_gen(noise_c, noise_d)

                    fake_c_seq, _ = c_vae.reconstruct_decoder(fake_z_c)
                    fake_d_seq, _ = d_vae.reconstruct_decoder(fake_z_d)

                    c_gen_eval.append(fake_c_seq.cpu().numpy())
                    d_gen_eval.append(fake_d_seq.cpu().numpy())
```

#### B. Đồng bộ hóa luồng sinh dữ liệu trong Checkpoint Visualization:
Tiếp tục trong `neo_m3gan/trainer.py`, đi tới block lưu checkpoint trực quan hóa cuối Epoch (khoảng dòng 427-446) và cập nhật thành:

```python
            with torch.no_grad():
                for _ in range(len(dataloader)):
                    noise_c = torch.randn(config['batch_size'], time_steps, noise_dim, device=device)
                    noise_d = torch.randn(config['batch_size'], time_steps, noise_dim, device=device)

                    # ROOT FIX: Đồng bộ hóa luồng sinh ảnh/NPZ với joint_gen
                    fake_z_c, fake_z_d = joint_gen(noise_c, noise_d)

                    fake_c_seq, _ = c_vae.reconstruct_decoder(fake_z_c)
                    fake_d_seq, _ = d_vae.reconstruct_decoder(fake_z_d)

                    c_gen_data.append(fake_c_seq.cpu().numpy())
                    d_gen_data.append(fake_d_seq.cpu().numpy())
```

#### C. Áp dụng KL Annealing để tránh Posterior Collapse ở VAE pretraining:
Tìm vòng lặp tiền huấn luyện VAE ở Phase 1 (khoảng dòng 104-134) và chỉnh sửa để tính factor annealing:

```python
        for epoch in range(config['num_pre_epochs']):
            # Tính toán hệ số nhân trọng số KL tăng dần tuyến tính từ 0.0 -> 1.0 trong 100 epoch đầu tiên
            kl_anneal_factor = min(1.0, epoch / 100.0)
            c_real_lst, c_rec_lst, d_real_lst, d_rec_lst = [], [], [], []
            epoch_total_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Pretrain Epoch [{epoch + 1}/{config['num_pre_epochs']}]", leave=True)

            for continuous_x, discrete_x in pbar:
                continuous_x = torch.clamp(torch.nan_to_num(continuous_x.to(device), nan=0.0), 0.0, 1.0)
                discrete_x = torch.clamp(torch.nan_to_num(discrete_x.to(device), nan=0.0), 0.0, 1.0)

                optimizer_VAE_pre.zero_grad()

                with torch.amp.autocast(device.type, enabled=config.get('use_amp', False)):
                    c_rec, _, c_mu, c_logvar, c_z = c_vae(continuous_x)
                    d_rec, d_logits, d_mu, d_logvar, d_z = d_vae(discrete_x)

                    loss_c_rec = F.mse_loss(c_rec, continuous_x)
                    loss_d_rec = F.binary_cross_entropy_with_logits(d_logits, discrete_x)

                    loss_c_kl = kl_divergence(c_mu, c_logvar)
                    loss_d_kl = kl_divergence(d_mu, d_logvar)

                    c_z_flat = c_z.view(c_z.size(0), -1)
                    d_z_flat = d_z.view(d_z.size(0), -1)
                    loss_contrastive = nt_xent_loss(c_z_flat, d_z_flat)
                    loss_matching = F.mse_loss(c_z, d_z)

                    # Áp dụng kl_anneal_factor vào thành phần KL Loss
                    total_vae_loss = (config['alpha_re'] * (loss_c_rec + loss_d_rec) +
                                      (config['alpha_kl'] * kl_anneal_factor) * (loss_c_kl + loss_d_kl) +
                                      config['alpha_ct'] * loss_contrastive +
                                      config['alpha_mt'] * loss_matching)
```

---

### Bước 3: Cập Nhật Siêu Tham Số Mặc Định (`neo_m3gan/main.py`)

Áp dụng cài đặt tiêu chuẩn toán học của WGAN-GP và cơ chế TTUR (D học nhanh hơn G).

**Vị trí sửa:** Tìm đoạn khai báo `d_rounds` và `d_lr` (khoảng dòng 119-125) trong `neo_m3gan/main.py` và cập nhật các giá trị default:

```python
    # Thay đổi từ default=2 lên default=5 để Critic huấn luyện đủ tối ưu
    parser.add_argument('--d_rounds', type=int, default=5)
    parser.add_argument('--g_rounds', type=int, default=1)
    parser.add_argument('--v_rounds', type=int, default=1)
    parser.add_argument('--v_lr_pre', type=float, default=0.0005)
    parser.add_argument('--v_lr', type=float, default=0.0001)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    # Tăng từ default=0.0001 lên default=0.0003 để thực hiện cơ chế TTUR
    parser.add_argument('--d_lr', type=float, default=0.0003)
```

---

## 3. Cách Kiểm Thử Hiệu Quả Cải Tiến

Sau khi bạn áp dụng các đoạn mã trên, hãy chạy huấn luyện thử nghiệm:
```bash
python3 neo_m3gan/main.py --dataset Mimic3 --num_epochs 200 --num_pre_epochs 300
```

**Các chỉ số bạn cần quan sát trong nhật ký/biểu đồ để thấy sự cải tiến:**
1.  **Chỉ số Continuous MMD:** Sẽ giảm sâu và ổn định hơn hẳn (không có hiện tượng MMD bất ngờ vọt lên do sụp đổ phân phối liên tục).
2.  **Continuous Feature Correlation Error (Pearson Correlation):** Sai số tương quan ma trận đặc trưng liên tục sẽ tiệm cận về mức thấp (gần khớp hoàn toàn với thực tế).
3.  **Trực quan hóa PDF mẫu sinh (Visualisation):** Các đặc trưng liên tục (như các đường biểu diễn Vital Signs) sẽ trải rộng đa dạng theo các hình dáng khác nhau, thay vì tất cả các bệnh nhân nhân tạo đều có chung một đường nằm ngang hoặc giống hệt nhau.
