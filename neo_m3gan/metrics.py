import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import warnings


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


def mmd_rbf(real_c, fake_c):
    """
    Calculates Maximum Mean Discrepancy (MMD) using an RBF kernel
    tuned via the Median Heuristic for high-dimensional timeseries.
    """
    from sklearn.metrics.pairwise import euclidean_distances

    X_flat = real_c.reshape(real_c.shape[0], -1)
    Y_flat = fake_c.reshape(fake_c.shape[0], -1)

    # Median Heuristic to find the proper RBF bandwidth (gamma)
    dists = euclidean_distances(X_flat, X_flat)
    sigma = np.median(dists[dists > 0])

    # Prevent division by zero if all data is identical
    if sigma == 0:
        sigma = 1.0

    gamma = 1.0 / (2.0 * (sigma ** 2))

    XX = rbf_kernel(X_flat, X_flat, gamma)
    YY = rbf_kernel(Y_flat, Y_flat, gamma)
    XY = rbf_kernel(X_flat, Y_flat, gamma)

    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd


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
    mmd_score = mmd_rbf(real_c, fake_c)
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