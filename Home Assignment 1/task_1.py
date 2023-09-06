import numpy as np
import matplotlib.pyplot as plt

def knn(training_points, training_labels, test_point, test_label):
    distances = np.diag(np.dot(training_points - test_point, (training_points - test_point).T))
    sorted_indices = np.argsort(distances)
    sorted_labels = training_labels[sorted_indices]
    cumulative_errors = np.sign(np.cumsum(sorted_labels)) != np.sign(test_label)
    return cumulative_errors

data_matrix = np.loadtxt("MNIST-5-6-Subset/MNIST-5-6-Subset.txt").reshape(1877, 784)
labels = np.loadtxt("MNIST-5-6-Subset/MNIST-5-6-Subset-Labels.txt")
labels = np.where(labels == 5, -1, 1)

# Using the first 50 training points and labels
training_points = data_matrix[:50]
training_labels = labels[:50]

m = 50
n_values = [10, 20, 40, 80]
validation_errors = []

# Iterate through different n values
for n in n_values:
    # Create validation sets
    validation_sets = []
    for i in range(5):
        start_idx = m + (i * n) + 1
        end_idx = m + ((i + 1) * n)
        validation_set_indices = range(start_idx, end_idx + 1)
        validation_sets.append(validation_set_indices)

    # Initialize an array to store validation errors for each validation set
    validation_errors_for_n = []

    # Iterate through each validation set
    for validation_indices in validation_sets:
        validation_errors_for_set = []

        # Iterate through each K value
        for k in range(1, m + 1):
            total_errors = 0

            # Iterate through each test point in the validation set
            for test_idx in validation_indices:
                test_point = data_matrix[test_idx]
                test_label = labels[test_idx]

                # Calculate errors for the current K value using knn function
                errors = knn(data_matrix[:m], labels[:m], test_point, test_label)
                total_errors += errors[k - 1]

            # Calculate validation error for the current K value
            validation_error = total_errors / len(validation_indices)
            validation_errors_for_set.append(validation_error)

        validation_errors_for_n.append(validation_errors_for_set)

    validation_errors.append(validation_errors_for_n)

# Plot the validation error curves for different n values
for i, n in enumerate(n_values):
    plt.figure(figsize=(8, 5))
    for j, validation_error_curve in enumerate(validation_errors[i], start=1):
        plt.plot(range(1, m + 1), validation_error_curve, label=f"Validation Set {j}")
    plt.xlabel("K")
    plt.ylabel("Validation Error")
    plt.title(f"Validation Error as a Function of K for n = {n}")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.ylim(0, 1)
    plt.xlim(1, 50)
    plt.show()

# Calculate the variance of validation errors for each K value and each n value
variance_per_n = []

for i, n in enumerate(n_values):
    variances_for_n = []
    for k in range(m):
        validation_errors_for_k = [validation_errors[i][j][k] for j in range(5)]
        variance_for_k = np.var(validation_errors_for_k)
        variances_for_n.append(variance_for_k)
    variance_per_n.append(variances_for_n)

# Plot the variance of validation errors for different n values as a function of K
for i, n in enumerate(n_values):
    plt.plot(range(1, m + 1), variance_per_n[i], label=f"n={n}")

plt.xlabel("K")
plt.ylabel("Variance of Validation Errors")
plt.title("Variance of Validation Errors vs. K for Different n")
plt.grid(alpha=0.2)
plt.legend()
plt.xlim(1, 50)
plt.show()