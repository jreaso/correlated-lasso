import numpy as np
from sklearn.linear_model import Lasso


def lasso_prediction_error(X, Y, beta_0, lambda_values):
    prediction_errors = []
    n, p = X.shape

    for lambda_ in lambda_values:
        lasso_model = Lasso(alpha=lambda_ / (2 * n), fit_intercept=False)
        # scale lambda to match objective function used by sklearn Lasso

        lasso_model.fit(X, Y)

        beta_hat = lasso_model.coef_
        prediction_errors.append(np.sum((X @ (beta_hat - beta_0)) ** 2))

    return prediction_errors
