import numpy as np
import matplotlib.pyplot as plt

# Number of repetitions and sample size
n_reps = 1000000
sample_size = 20

# Initialize an array to store the empirical frequencies
empirical_frequencies = []

# Generate random variables and compute sample means for each repetition
for _ in range(n_reps):
    sample = np.random.binomial(1, 0.5, sample_size)  # Generate 20 Bernoulli random variables
    sample_mean = np.mean(sample)  # Compute the sample mean
    empirical_frequencies.append(sample_mean)

# Define a list of alpha values
alpha_values = np.arange(0.5, 1, 0.05)

# Initialize an array to store the empirical frequencies for each alpha
alpha_frequencies = []

# Calculate the empirical frequencies for each alpha
for alpha in alpha_values:
    count = sum(1 for freq in empirical_frequencies if freq >= alpha)
    alpha_frequencies.append(count / n_reps)

# Plot the empirical frequencies
plt.plot(alpha_values, alpha_frequencies, marker='o', linestyle='-', color='b')
plt.xlabel('Alpha')
plt.ylabel('Empirical Frequency')
plt.title('Empirical Frequency of Sum of 20 Bernoulli Variables')
plt.grid(True)
plt.show()
