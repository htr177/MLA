import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import binom

# Number of repetitions and sample size
num_repetitions = 1000000
sample_size = 20
coin_bias = 0.5
seed = 42
np.random.seed(seed)

# Initialize an array to store the empirical frequencies
empirical_frequencies = []

# Generate random variables and compute sample means for each repetition
for _ in range(num_repetitions):
    sample = np.random.binomial(1, 0.5, sample_size)  # Generate 20 Bernoulli random variables
    sample_mean = np.mean(sample)  # Compute the sample mean
    empirical_frequencies.append(sample_mean)

# Define a list of alpha values
alpha_values = np.arange(0.5, 1.05, 0.05)

# Initialize an array to store the empirical frequencies for each alpha
alpha_frequencies = []

# Calculate the empirical frequencies for each alpha
for alpha in alpha_values:
    count = sum(1 for freq in empirical_frequencies if freq >= alpha)
    alpha_frequencies.append(count / num_repetitions)

# Calculate the Markov bounds for each alpha using coin_bias
markov_bounds = []

for alpha in alpha_values:
    markov_bound = coin_bias / alpha
    markov_bounds.append(markov_bound)

# Calculate the Chebyshev bounds for each alpha using coin_bias
chebyshev_bounds = []

for alpha in alpha_values:
    chebyshev_bound = ((coin_bias * (1 - coin_bias)) / (alpha * sample_size)) / ((alpha - coin_bias) ** 2)
    chebyshev_bounds = [min(1, bound) for bound in chebyshev_bounds]
    chebyshev_bounds.append(chebyshev_bound)
    

# Calculate the Hoeffding bounds for each alpha
hoeffding_bounds = []

for alpha in alpha_values:
    hoeffding_bound = np.e ** (-2 * sample_size * (alpha - coin_bias) ** 2)
    hoeffding_bounds.append(hoeffding_bound)

# Plot the Markov bounds, Chebyshev bounds, and Hoeffding bounds
plt.plot(alpha_values, markov_bounds, linestyle='-', color='r', label="Markov's Bound")
plt.plot(alpha_values, chebyshev_bounds, linestyle='-', color='g', label="Chebyshev's Bound")
plt.plot(alpha_values, hoeffding_bounds, linestyle='-', color='purple', label="Hoeffding's Bound")

# Plot the empirical frequencies as vertical bars
plt.vlines(alpha_values, 0, alpha_frequencies, color='b', linewidth=4, label='Empirical Frequency')

plt.xlabel('Alpha')
plt.ylabel('Frequency/Probability')
plt.title('Empirical Frequency and Bounds')
plt.legend()
plt.grid(False)
plt.ylim((0.0, 1.1))
plt.xlim((0.49, 1.02))
plt.xticks([i for i in np.arange(0.5, 1.05, 0.05)])

# Format the Y-axis as percentages
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

plt.show()


####################################################################################################

# Parameters for the binomial distribution
n = 20
p = 0.5

# Define the alpha values
alpha_1 = 1
alpha_0_95 = 0.95

# Calculate the probabilities using the binomial CDF
probability_alpha_1 = 1 - binom.cdf(n * alpha_1 - 1, n, p)
probability_alpha_0_95 = 1 - binom.cdf(n * alpha_0_95 - 1, n, p)

print(f'P(1/20 * sum(X_i) >= {alpha_1}) = {probability_alpha_1:.20f}')
print(f'P(1/20 * sum(X_i) >= {alpha_0_95}) = {probability_alpha_0_95:.20f}')

####################################################################################################