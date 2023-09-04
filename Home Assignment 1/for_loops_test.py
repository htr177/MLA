import numpy as np
import matplotlib.pyplot as plt

def knn(training_points, training_labels, test_point, test_label):
    distances = np.sum((training_points - test_point)**2, axis=1)
    sorted_indices = np.argsort(distances)
    sorted_labels = training_labels[sorted_indices]
    cumulative_errors = np.sign(np.cumsum(sorted_labels)) != np.sign(test_label)
    return cumulative_errors

def create_validation_sets(m, n_values):
    validation_sets = []
    for n in n_values:
        for i in range(5):
            start_idx = m + (i * n) + 1
            end_idx = m + ((i + 1) * n)
            validation_set_indices = range(start_idx, end_idx + 1)
            validation_sets.append(validation_set_indices)
    return validation_sets

def calculate_validation_errors(m, n_values, training_points, training_labels, data_matrix, labels):
    validation_errors = []

    for n in n_values:
        validation_sets = create_validation_sets(m, [n])

        validation_errors_for_n = []
        
        for validation_indices in validation_sets:
            validation_errors_for_set = []
            
            for k in range(1, m + 1):
                total_errors = 0
                
                for test_idx in validation_indices:
                    test_point = data_matrix[test_idx]
                    test_label = labels[test_idx]
                    errors = knn(training_points[:m], training_labels[:m], test_point, test_label)
                    total_errors += errors[k - 1]

                validation_error = total_errors / len(validation_indices)
                validation_errors_for_set.append(validation_error)
                
            validation_errors_for_n.append(validation_errors_for_set)
        
        validation_errors.append(validation_errors_for_n)
    
    return validation_errors

def plot_validation_error_scatter(n_values, m, validation_errors):
    plt.clf()
    plt.scatter([i for i in range(1, m + 1)], validation_errors[0][0], color='red', label='n = 10')
    plt.scatter([i for i in range(1, m + 1)], validation_errors[1][0], color='blue', label='n = 20')
    plt.scatter([i for i in range(1, m + 1)], validation_errors[2][0], color='green', label='n = 40')
    plt.scatter([i for i in range(1, m + 1)], validation_errors[3][0], color='orange', label='n = 80')
    plt.xlabel('K')
    plt.ylabel('Validation Error')
    plt.title('Validation Error vs. K for Different n Values')
    plt.legend()
    plt.show()

        

# Load the dataset and set parameters
data_matrix = np.loadtxt("MNIST-5-6-Subset/MNIST-5-6-Subset.txt").reshape(1877, 784)
labels = np.loadtxt("MNIST-5-6-Subset/MNIST-5-6-Subset-Labels.txt")
labels = np.where(labels == 5, -1, 1)

# Using the first 50 training points and labels
training_points = data_matrix[:50]
training_labels = labels[:50]

m = 50
n_values = [10, 20, 40, 80]

# Calculate validation errors
validation_errors = calculate_validation_errors(m, n_values, training_points, training_labels, data_matrix, labels)

# Plot validation error curves
plot_validation_error_scatter(n_values, m, validation_errors)