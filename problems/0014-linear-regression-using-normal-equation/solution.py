import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X = np.array(X)
	y = np.array(y)
	X_T = np.transpose(X)
	X_T_X = X_T @ X
	X_T_y = X_T @ y
	X_inv = np.linalg.inv(X_T_X)
	theta = X_inv @ X_T_y
	
	return np.round(theta, 4).tolist()