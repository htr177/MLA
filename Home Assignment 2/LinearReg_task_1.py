import numpy as np
import matplotlib.pyplot as plt

# Read the data from the file
data = np.loadtxt('PCB.dt')

X = data[:, 0]
Y = data[:, 1]

# Create a column with 1s for the intercept
X = np.c_[X, np.ones(X.shape[0])]

# Calculate the coefficients using the normal equation
coefficients = np.linalg.inv(X.T @ X) @ X.T @ Y
print(coefficients)

# Plot the data points
plt.scatter(X[:, 0], Y, marker='o', color='b', label='Data Points')

# Plot the regression line
# plt.plot(X[:, 0], X @ coefficients, linestyle='-', color='r', label='Regression Line')
#plt.show()

# Build a non-linear model with coefficient from the linear regression
Y_log = np.log(Y)

# Find coefficents a and b for the non-linear model
coefficients_log = np.linalg.inv(X.T @ X) @ X.T @ Y_log
print(coefficients_log)

# Use these new coefficients to plot the non-linear function (exp(a*x + b))
exp = np.exp(coefficients_log)
print("Log coeffs", exp)

def h(x):
    return np.exp(coefficients_log[0] * x + coefficients_log[1])

preds = h(X[:, 0])

# calculate mse
mse = np.mean((preds - Y) ** 2)
print("MSE", mse)

# Plot the data and the model output
x = np.linspace(0, 14, 100)
y = coefficients_log[0] * x + coefficients_log[1]
plt.figure()
plt.plot(X[:, 0], Y_log, 'o')
plt.xlabel('Data')
plt.ylabel('Log of Data')
plt.plot(x, y, 'r-')
plt.show()


# Compute the coefficient of determination
R_2 = 1 - mse / np.var(Y)
print("R2", R_2)

