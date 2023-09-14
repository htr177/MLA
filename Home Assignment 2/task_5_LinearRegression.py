import numpy as np
import matplotlib.pyplot as plt

# Read the data from the file
data = np.loadtxt("Home Assignment 2/PCB.dt")

X = data[:, 0]
Y = data[:, 1]

# Create a column with 1s for the intercept
X = np.c_[X, np.ones(X.shape[0])]

# Calculate the coefficients using the normal equation
coefficients = np.linalg.inv(X.T @ X) @ X.T @ Y
print("Linear model coefficients", coefficients)

# Plot the data points
plt.scatter(X[:, 0], Y, marker='o', color='b', label='Data Points')

# Plot the regression line
plt.plot(X[:, 0], X @ coefficients, linestyle='-', color='r', label='Regression Line')
plt.xlabel('Age')
plt.ylabel('PCB Concentration')
plt.title('Linear Regression')
plt.legend()

# Build a non-linear model with coefficient from the linear regression
Y_log = np.log(Y)

# Find coefficients a and b for the non-linear model
coefficients_log = np.linalg.inv(X.T @ X) @ X.T @ Y_log
print("Parameters (Y_log): ", coefficients_log)

def h(x):
    return np.exp(coefficients_log[0] * x + coefficients_log[1])

preds = h(X[:, 0])

# Calculate mse
mse = np.mean((preds - Y) ** 2)
print("MSE:", mse)

# Compute the coefficient of determination
R_2 = 1 - mse / np.var(Y)
print("R2", R_2)

# Plot the data and the model output
x = np.linspace(0, 14, 100)
y = coefficients_log[0] * x + coefficients_log[1]
plt.figure()
plt.plot(X[:, 0], Y_log, 'o', label='Log of PCB')
plt.xlabel('Age')
plt.ylabel('Log of PCB Concentration')
plt.xlim((0, 13))
plt.ylim((-1, 5))
plt.title('Affine Linear Model')
plt.plot(x, y, 'r-', label='Non-linear Model')
plt.legend()
plt.show()

# Build a non-linear model with coefficient from the linear regression
X.T[0] = np.sqrt(X.T[0])
Y_log = np.log(Y)

# Find coefficients a and b for the non-linear model
coefficients_log = np.linalg.inv(X.T @ X) @ X.T @ Y_log
print("Parameters (Y_log): ", coefficients_log)

# Define the non-linear model h(x) = exp(a√x + b)
a_sqrt = coefficients_log[0]
b_sqrt = coefficients_log[1]

def h_sqrt(x):
    return np.exp(a_sqrt * (x) + b_sqrt)

# Calculate the predictions
preds_sqrt = h_sqrt(X[:, 0])

# Calculate the MSE for the non-linear model
mse_sqrt = np.mean((preds_sqrt - Y) ** 2)
print("MSE sqrt", mse_sqrt)

# Compute the coefficient of determination
R_2_sqrt = 1 - mse_sqrt / np.var(Y)
print("R2 sqrt", R_2_sqrt)

# Plot the data and the model output
x_values = np.linspace(0, 14, 100)
y_values = a_sqrt * np.sqrt(x_values) + b_sqrt
X.T[0] = (X.T[0])**2

plt.figure()
plt.plot(X[:, 0], Y_log, 'o', label='Log of Data')
plt.title('Log of Data vs. Non-linear Model (sqrt of x)')
plt.xlabel('Age')
plt.ylabel('Log of PCB Concentration')
plt.plot(x_values, y_values, 'r-', label='Non-linear Model')
plt.xlim((0, 13))
plt.ylim((-1, 4))
plt.legend()
plt.show()