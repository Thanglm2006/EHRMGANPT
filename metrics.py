import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import warnings
import torch


def discrete_probability_rmse(real_d, fake_d):
    """
    Calculates the Root Mean Square Error (RMSE) between the
    dimension-wise probabilities of the real and synthetic discrete data.
    """
    # Calculate probability of each code occurring across all samples and timesteps
    prob_real = np.mean(real_d, axis=(0, 1))
    prob_fake = np.mean(fake_d, axis=(0, 1))

    rmse = np.sqrt(np.mean((prob_real - prob_fake) ** 2))
    return rmse


def _mix_rbf_kernel(X, Y, sigmas, wts=None):
    if wts is None:
        wts = [1.0] * sigmas.shape[0]

    # Flatten dimensions similar to tf.tensordot(axes=[[1, 2], [1, 2]])
    X_flat = X.reshape(X.shape[0], -1)
    Y_flat = Y.reshape(Y.shape[0], -1)

    XX = torch.matmul(X_flat, X_flat.t())
    XY = torch.matmul(X_flat, Y_flat.t())
    YY = torch.matmul(Y_flat, Y_flat.t())

    X_sqnorms = torch.diag(XX)
    Y_sqnorms = torch.diag(YY)

    K_XX, K_XY, K_YY = 0., 0., 0.
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma ** 2)
        K_XX += wt * torch.exp(-gamma * (-2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)))
        # X_sqnorms is (batch_x,), shape (m, 1) and Y_sqnorms is (batch_y,), shape (1, n)
        K_XY += wt * torch.exp(-gamma * (-2 * XY + X_sqnorms.unsqueeze(1) + Y_sqnorms.unsqueeze(0)))
        K_YY += wt * torch.exp(-gamma * (-2 * YY + Y_sqnorms.unsqueeze(1) + Y_sqnorms.unsqueeze(0)))

    return K_XX, K_XY, K_YY, sum(wts)


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.shape[0]
    n = K_YY.shape[0]

    if biased:
        mmd2 = (torch.sum(K_XX) / (m * m)
                + torch.sum(K_YY) / (n * n)
                - 2 * torch.sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = torch.trace(K_XX)
            trace_Y = torch.trace(K_YY)

        mmd2 = ((torch.sum(K_XX) - trace_X) / (m * (m - 1))
                + (torch.sum(K_YY) - trace_Y) / (n * (n - 1))
                - 2 * torch.sum(K_XY) / (m * n))

    return mmd2


def mix_rbf_mmd2(X, Y, sigmas=None, wts=None, biased=True):
    if sigmas is None:
        sigmas = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0], device=X.device)
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


def max_mean_discrepancy(real_data, syn_data, bandwidths=None):
    if not isinstance(real_data, torch.Tensor):
        X = torch.tensor(real_data, dtype=torch.float32)
    else:
        X = real_data.float()
        
    if not isinstance(syn_data, torch.Tensor):
        Y = torch.tensor(syn_data, dtype=torch.float32)
    else:
        Y = syn_data.float()

    if bandwidths is not None:
        if not isinstance(bandwidths, torch.Tensor):
            bandwidths = torch.tensor(bandwidths, dtype=torch.float32, device=X.device)

    # Move Y to X device in case they differ
    Y = Y.to(X.device)
    
    mmd2_value = mix_rbf_mmd2(X, Y, sigmas=bandwidths, biased=True) 

    # Prevent nan from sqrt of tiny negative numbers due to floating point precision
    mmd2_value = torch.clamp(mmd2_value, min=0.0) 
    
    return torch.sqrt(mmd2_value).item()


def pearson_correlation_error(real_data, fake_data):
    """
    Calculates the Mean Absolute Error between the Pearson correlation
    matrices of the real and synthetic features.
    """
    # Flatten temporal dimension to calculate overall feature-wise correlation
    # shape: (num_samples * time_steps, features)
    real_flat = real_data.reshape(-1, real_data.shape[2])
    fake_flat = fake_data.reshape(-1, fake_data.shape[2])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Calculate correlation matrices (features x features)
        corr_real = np.corrcoef(real_flat, rowvar=False)
        corr_fake = np.corrcoef(fake_flat, rowvar=False)

    # If a feature is entirely 0, correlation becomes NaN. Convert to 0.
    corr_real = np.nan_to_num(corr_real, nan=0.0)
    corr_fake = np.nan_to_num(corr_fake, nan=0.0)

    # Calculate mean absolute error between the matrices
    error = np.mean(np.abs(corr_real - corr_fake))
    return error


def evaluate_all(real_c, fake_c, real_d, fake_d):
    print("\n" + "=" * 40)
    print(" EHR-M-GAN EVALUATION METRICS")
    print("=" * 40)

    # 1. Continuous MMD
    mmd_score = max_mean_discrepancy(real_c, fake_c)
    print(f"Continuous MMD (Lower is better):      {mmd_score:.5f}")

    # 2. Discrete Probability RMSE
    rmse_score = discrete_probability_rmse(real_d, fake_d)
    print(f"Discrete Prob RMSE (Lower is better):  {rmse_score:.5f}")

    # 3. Continuous Feature Correlation Error
    corr_err_c = pearson_correlation_error(real_c, fake_c)
    print(f"Continuous Corr Error (Lower is better): {corr_err_c:.5f}")

    # 4. Discrete Feature Correlation Error
    corr_err_d = pearson_correlation_error(real_d, fake_d)
    print(f"Discrete Corr Error (Lower is better):   {corr_err_d:.5f}")
    print("=" * 40 + "\n")

    return {
        'mmd': float(mmd_score),
        'rmse': float(rmse_score),
        'corr_c': float(corr_err_c),
        'corr_d': float(corr_err_d)
    }