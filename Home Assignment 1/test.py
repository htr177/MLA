import numpy as np
import matplotlib.pyplot as plt

# Load the data from MNIST-5-6-Subset.txt
# Change the path as needed

# data_file_path = "MNIST-5-6-Subset/MNIST-5-6-Subset.txt"
# data_matrix = np.loadtxt(data_file_path).reshape(1877, 784)
# # Load the labels from MNIST-5-6-Labels.txt
# # Change the path as needed
# labels_file_path = "MNIST-5-6-Subset/MNIST-5-6-Subset-Labels.txt"
# labels = np.loadtxt(labels_file_path)
# # Assuming you want to visualize the first image
# # Change the index as needed
# image_index = 0
# image_data = data_matrix[image_index]
# selected_label = int(labels[image_index])
# # Visualize the image using Matplotlib
# # We transpose the image to make the number look upright.
# plt.imshow(image_data.reshape(28,28).transpose(1,0), cmap='gray')
# plt.title(f"Label: {selected_label}")
# plt.axis('off') # Turn off axis
# plt.show()

# #######################################################################################################
# # Setting up a figure with axis labels, legend and title
# #######################################################################################################

# # Dummy data, x and y
# x = np.arange(0, 20.1, 0.1)
# y = np.sin(x) + np.random.normal(0, 0.2, len(x))
# some_parameter = 54
# # Initialise figure (fig) and axis (ax)
# fig, ax = plt.subplots(figsize=(8,5))
# # Plot in axis, add label to data
# ax.plot(x, y, label='Dummy data') # (*)
# # Set labels and title
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_title(f'Dummy data with some parameter = {some_parameter}')
# # Add grid
# ax.grid(alpha=0.2)
# # Set axes limits
# ax.set_ylim(-2,2)
# # Add legend (remember to label the data as shown above (*))
# ax.legend()
# # Show plot
# plt.show()
# # Save plot to some local path
# fig.savefig('validation_err.png')


# #########################################################################################################
# # Using vector operations with Numpy
# #########################################################################################################


# # Say we have a data matrix with dimension (50, 10)
# data_matrix = np.random.rand(50, 10)
# print('data_matrix shape:', data_matrix.shape)
# # .. and we want to subtract from all of its columns a vector of dimension (10)
# some_vector = np.random.rand(10)
# print('some_vector shape:', some_vector.shape)
# # Instead of looping through the data matrix and subtracting like so,
# result_loop = np.zeros_like(data_matrix)
# for i,column in enumerate(data_matrix):
#     result_loop[i] = column - some_vector
# print('result_loop shape:', result_loop.shape)

# # We can use vector operations to greatly improve the speed, at which we achieve the same result. 
# # The essential action involves expanding the dimensions of "some_vector", aligning it with the dimensions of the "data_matrix." 
# # np.newaxis accomplishes this by encapsulating the original data with ": ", while simultaneously creating a new dimension.
# some_vector_new = some_vector[np.newaxis, :]
# print('some_vector shape after expansion:', some_vector_new.shape)

# # Now we can subtract some_vector simply like this
# result_vector = data_matrix - some_vector_new
# print('result_vector shape:', result_vector.shape)

# # Assert that the two results are equal
# print('result_loop == result_vector:', np.all(result_loop == result_vector))

# # We can easily check how large of a speedup we achieve
# # by using the time package
# from time import time
# loop_time = []
# vector_time = []
# for _ in range(250):
#     # For loop
#     t = time()
#     for i,column in enumerate(data_matrix):
#         result_loop[i] = column - some_vector
#     loop_time.append(t - time())
#     # Vector operation
#     t = time()
#     result_vector = data_matrix - some_vector[np.newaxis, :]
#     vector_time.append(t - time())

# print(f'Speed up: {(np.mean(loop_time) / np.mean(vector_time)):1.3f}')


# #########################################################################################################
# # Other useful Numpy functions: cumsum, sort and argsort
# #########################################################################################################

# # Creating an example array
# data = np.array([5, 2, 8, 1, 6])

# # 1)
# # Calculating cumulative sum using cumsum
# cumulative_sum = np.cumsum(data)
# print("Original data:", data)
# print("Cumulative sum:", cumulative_sum)
# # Documentation for np.cumsum: https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html

# # 2)
# # Sorting the array using sort
# sorted_data = np.sort(data)
# print("\nOriginal data:", data)
# print("Sorted data:", sorted_data)
# # Documentation for np.sort: https://numpy.org/doc/stable/reference/generated/numpy.sort.html

# # 3)
# # Getting indices that would sort the array using argsort
# sorted_indices = np.argsort(data)
# print("\nOriginal data:", data)
# print("Sorted indices:", sorted_indices)
# # Documentation for np.argsort: https://numpy.org/doc/stable/reference/generated/numpy.argsort.html

# # 4)
# # Accessing elements in sorted order using sorted indices
# sorted_data_using_indices = data[sorted_indices]
# print("\nOriginal data:", data)
# print("Sorted data using indices:", sorted_data_using_indices)

# #########################################################################################################


"""
Implement a Python function knn(training_points, training_labels,
test_point, test_label) that takes as input a d × m matrix of training
points training_points, where m is the number of training points and
d is the dimension of each point (d = 784 in the case of digits), a vector
training_labels of the corresponding m training labels, a d-dimensional
vector text_point representing one test point, and its label test_label
∈ {−1, 1} (you will need to convert the labels from {5, 6} to {−1, 1}).
The function should return a binary vector of length m, where each element represents the error of K-NN for the corresponding value of K for
K ∈ {1, . . . , m}. Include a printout of your implementation of the
function in the report. (Only this function, not all of your code, the complete code should be included in the .zip file.) Ideally, the function should
have no for-loops, check the practical advice at the end of the question.
"""

def knn(training_points, training_labels, test_point, test_label):
    distances = np.sum((training_points - test_point)**2, axis=1)
    sorted_indices = np.argsort(distances)
    sorted_labels = training_labels[sorted_indices]
    cumulative_errors = np.sign(np.cumsum(sorted_labels)) != np.sign(test_label)
    return cumulative_errors

# Load data
data_matrix = np.loadtxt("MNIST-5-6-Subset/MNIST-5-6-Subset.txt").reshape(1877, 784)
# data_matrix = data_matrix.T
labels = np.loadtxt("MNIST-5-6-Subset/MNIST-5-6-Subset-Labels.txt")
labels = np.where(labels == 5, -1, 1)


# Using the first 50 training points and labels
training_points = data_matrix[:50]
training_labels = labels[:50]

# Test point and label for the specific example
test_point = data_matrix[151]
test_label = labels[151]

for k in range(1, 51):
    errors = knn(training_points, training_labels, test_point, test_label)
    num_errors = errors[k-1].astype(int)
    print(f"K = {k}, Number of Errors = {num_errors}")


################################################################################################################

# Define constants
m = 50  # Number of training points
n_values = [10, 20, 40, 80]  # Sizes of validation sets

# Iterate through different n values
for n in n_values:
    validation_sets = []
    validation_errors = []

    # Create validation sets
    for i in range(5):
        start_idx = m + (i * n) + 1
        end_idx = m + ((i + 1) * n)
        validation_set_indices = range(start_idx, end_idx + 1)
        validation_sets.append(validation_set_indices)

    # Calculate validation errors for each validation set and K
    for validation_indices in validation_sets:
        validation_errors_for_set = []

        for K in range(1, m + 1):
            num_errors = 0

            for test_idx in validation_indices:
                test_point = data_matrix[test_idx]
                test_label = labels[test_idx]
                error = knn(training_points, training_labels, test_point, test_label)
                num_errors += error

            validation_error = num_errors / n
            validation_errors_for_set.append(validation_error)

        validation_errors.append(validation_errors_for_set)

    # Plot validation error curves for each validation set
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, validation_error_curve in enumerate(validation_errors, start=1):
        ax.plot(range(1, m + 1), validation_error_curve, label=f"Validation Set {i}")

    ax.set_xlabel("K")
    ax.set_ylabel("Validation Error")
    ax.set_title(f"Validation Error as a Function of K (n={n})")
    ax.grid(alpha=0.2)
    ax.legend()
    plt.show()
    # Save figure
    plt.savefig(f"validation_error_n_{n}.png")

