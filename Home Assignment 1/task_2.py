import numpy as np
import matplotlib.pyplot as plt

def knn(training_points, training_labels, test_point, test_label):
    distances = np.diag(np.dot(training_points - test_point, (training_points - test_point).T))
    sorted_indices = np.argsort(distances)
    sorted_labels = training_labels[sorted_indices]
    cumulative_errors = np.sign(np.cumsum(sorted_labels)) != np.sign(test_label)
    return cumulative_errors

data_matrix = np.loadtxt("MNIST-5-6-Subset/MNIST-5-6-Subset-Heavy-Corruption.txt").reshape(1877, 784)
labels = np.loadtxt("MNIST-5-6-Subset/MNIST-5-6-Subset-Labels.txt")
labels = np.where(labels == 5, -1, 1)

# Using the first 50 training points and labels
training_points = data_matrix[:50]
training_labels = labels[:50]


# Set n and m
n = 80
m = 50

# Create five validation sets
validation_sets = []
for i in range(5):
    start_idx = m + (i * n) + 1
    end_idx = m + ((i + 1) * n)
    validation_set_indices = range(start_idx, end_idx + 1)
    validation_sets.append(validation_set_indices)

# Initialize an array to store validation errors for each validation set
validation_errors = []

# Iterate through each validation set
for validation_indices in validation_sets:
    validation_errors_for_set = []


    total_errors = np.zeros(50)

    # Iterate through each test point in the validation set
    for test_idx in validation_indices:
        test_point = data_matrix[test_idx]
        test_label = labels[test_idx]

        # Calculate errors for the current K value using knn function
        errors = knn(data_matrix[:m], labels[:m], test_point, test_label)
        total_errors += errors/len(validation_indices)

    # Calculate validation error for the current K value
    validation_error = total_errors / len(validation_indices)
    validation_errors_for_set.append(validation_error)

validation_errors.append(validation_errors_for_set)

# Plot the validation error curves for different K values
for i, validation_error_curve in enumerate(validation_errors, start=1):
    plt.plot(range(1, m + 1), validation_error_curve, label=f"Validation Set {i}")

plt.xlabel("K")
plt.ylabel("Validation Error")
plt.title("Validation Error as a Function of K for n = 80 (Heavy Corruption)")
plt.grid(alpha=0.2)
plt.legend()
plt.show()

