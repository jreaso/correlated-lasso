import numpy as np

# from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# from utils import lasso_prediction_error
import sparse_lm as slm

lambda_values = np.linspace(0, 15, num=101)

mean_prediction_errors_algo1 = slm.prediction_error(
    n=20, p=40, s=4, lambda_values=lambda_values
)

mean_prediction_errors_algo2 = slm.prediction_error(
    n=20, p=40, s=4, lambda_values=lambda_values, correlated=True, eta=0.001
)

plt.figure(figsize=(8, 6))
plt.ylim(0, 20)
plt.plot(lambda_values, mean_prediction_errors_algo1, color="b", linewidth=1)
plt.plot(lambda_values, mean_prediction_errors_algo2, color="r", linewidth=1)
plt.xlabel("Lambda", fontsize=12)
plt.ylabel("Mean Prediction Error", fontsize=12)
plt.show()
