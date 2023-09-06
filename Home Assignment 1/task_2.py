import numpy as np
import matplotlib.pyplot as plt

def knn(training_points, training_labels, test_point, test_label):
    distances = np.diag(np.dot(training_points - test_point, (training_points - test_point).T))
    sorted_indices = np.argsort(distances)
    sorted_labels = training_labels[sorted_indices]
    cumulative_errors = np.sign(np.cumsum(sorted_labels)) != np.sign(test_label)
    return cumulative_errors

data_matrix = np.loadtxt("Home Assignment 1/MNIST-5-6-Subset/MNIST-5-6-Subset-Heavy-Corruption.txt").reshape(1877, 784)
labels = np.loadtxt("Home Assignment 1/MNIST-5-6-Subset/MNIST-5-6-Subset-Labels.txt")
labels = np.where(labels == 5, -1, 1)

# Using the first 50 training points and labels
training_points = data_matrix[:50]
training_labels = labels[:50]

# Set n and m
n_values = [80]
m = 50
validation_errors = []

# Create five validation sets
validation_sets = []
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
        total_errors = np.zeros(m)  # Initialize an array to store errors for each K value

        # Iterate through each test point in the validation set
        for test_idx in validation_indices:
            test_point = data_matrix[test_idx]
            test_label = labels[test_idx]

            # Calculate errors for the current K value using knn function
            errors = knn(data_matrix[:m], labels[:m], test_point, test_label)
            total_errors += errors

        # Calculate validation error for each K value
        validation_error_curve = total_errors / len(validation_indices)
        validation_errors_for_n.append(validation_error_curve)

    validation_errors.extend(validation_errors_for_n)

# Plot the validation error curves for different n values
for i, n in enumerate(n_values):
    plt.figure(figsize=(8, 5))
    for j in range(5):
        plt.plot(range(1, m + 1), validation_errors[i * 5 + j], label=f"Validation Set {j + 1}")
    plt.xlabel("K")
    plt.ylabel("Validation Error")
    plt.title(f"Validation Error as a Function of K for n = {n}")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.show()
