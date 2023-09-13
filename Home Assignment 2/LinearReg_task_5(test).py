import numpy as np
import matplotlib.pyplot as plt

# Read the data from the file
data = np.loadtxt("Home Assignment 2/PCB.dt")

X = data[:, 0]
Y = data[:, 1]

# Transform the input data (X) by taking the square root
X_transformed = np.sqrt(X)

# Create a new column with 1s for the intercept
X_transformed = np.c_[X_transformed, np.ones(X_transformed.shape[0])]

# Calculate the coefficients using the normal equation for the linear model
coefficients = np.linalg.inv(X_transformed.T @ X_transformed) @ X_transformed.T @ np.log(Y)

# Define the non-linear model h(x) = exp(a√x + b)
a_sqrt = coefficients[0]
b_sqrt = coefficients[1]

def h_sqrt(x):
    return np.exp(a_sqrt * np.sqrt(x) + b_sqrt)

# Calculate the model predictions
preds_sqrt = h_sqrt(X)

# Calculate mean squared error for the non-linear model
mse_sqrt = np.mean((preds_sqrt - Y) ** 2)
print("MSE for the non-linear model:", mse_sqrt)

# Compute the coefficient of determination
R_2_sqrt = 1 - mse_sqrt / np.var(Y)
print("R2 for the non-linear model:", R_2_sqrt)

# Plot the data and the model output
x_values = np.linspace(0, 14, 100)
y_values = h_sqrt(x_values)

plt.figure()
plt.plot(X, Y, 'o', label='Data')
plt.xlabel('Age')
plt.ylabel('PCB Concentration')
plt.plot(x_values, y_values, 'r-', label='Non-linear Model')
plt.legend()
plt.show()
