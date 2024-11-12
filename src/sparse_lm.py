import numpy as np
from utils import lasso_prediction_error


# Follows Algorithm 1 in Hebiria and Lederer (2013) paper
def simulate_simple_model(beta_0, n, p, sigma, Sigma):
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)
    X = X / np.sqrt(np.sum(X**2, axis=0) / n)  # normalize rows

    noise = np.random.normal(0, sigma, n)
    Y = X @ beta_0 + noise

    return (X, Y)


# Follows Algorithm 2 in Hebiria and Lederer (2013) paper
def simulate_correlated_model(beta_0, n, p, sigma, Sigma, eta):
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)

    # For each column j, add p-1 correlated copies
    X = np.hstack(
        [
            X,
            np.hstack(
                [
                    X[:, j].reshape(-1, 1)
                    + np.random.normal(loc=0, scale=eta, size=(n, p - 1))
                    for j in range(p)
                ]
            ),
        ]
    )

    X = X / np.sqrt(np.sum(X**2, axis=0) / n)  # normalize rows

    noise = np.random.normal(0, sigma, n)
    Y = X @ beta_0 + noise

    return (X, Y)


def prediction_error(
    n, p, s, lambda_values, sigma=1, rho=0, n_iter=20, correlated=False, eta=None
):
    beta_0 = np.zeros(p)
    beta_0[:s] = 1

    Sigma = np.full((p, p), rho)
    np.fill_diagonal(Sigma, 1)

    if not correlated:
        prediction_errors_array = np.array(
            [
                lasso_prediction_error(
                    *simulate_simple_model(beta_0, n, p, sigma, Sigma),
                    beta_0,
                    lambda_values,
                )
                for _ in range(n_iter)
            ]
        )
    else:
        beta_0 = np.pad(
            beta_0, (0, p**2 - p), mode="constant"
        )  # extend beta_0 to be zero on added variables
        prediction_errors_array = np.array(
            [
                lasso_prediction_error(
                    *simulate_correlated_model(beta_0, n, p, sigma, Sigma, eta),
                    beta_0,
                    lambda_values,
                )
                for _ in range(n_iter)
            ]
        )

    return np.mean(prediction_errors_array, axis=0)
