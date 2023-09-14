import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

def calculate_bounds_and_plot(coin_bias, alpha_start, alpha_end, alpha_step):
    # Number of repetitions and sample size
    num_repetitions = 1000000
    sample_size = 20
    seed = 42
    np.random.seed(seed)

    # Initialize an array to store the empirical frequencies
    empirical_frequencies = []

    # Generate random variables and compute sample means for each repetition
    sample = np.random.binomial(sample_size, coin_bias, num_repetitions)
    sample_mean = sample / sample_size
    empirical_frequencies.append(sample_mean)

    # Define a list of alpha values
    alpha_values = np.arange(alpha_start, alpha_end + alpha_step, alpha_step)

    # Initialize arrays to store bounds
    markov_bounds = []
    chebyshev_bounds = []
    hoeffding_bounds = []

    for alpha in alpha_values:
        # Calculate the Markov bounds for each alpha using coin_bias
        markov_bound = coin_bias / alpha
        markov_bounds.append(markov_bound)

        # Calculate the Chebyshev bounds for each alpha using coin_bias
        chebyshev_bound = ((coin_bias * (1 - coin_bias)) / (sample_size * (alpha - coin_bias) ** 2))
        chebyshev_bounds = [min(1, bound) for bound in chebyshev_bounds]
        chebyshev_bounds.append(chebyshev_bound)

        # Calculate the Hoeffding bounds for each alpha
        hoeffding_bound = np.e ** (-2 * sample_size * (alpha - coin_bias) ** 2)
        hoeffding_bounds.append(hoeffding_bound)

    # Plot the bounds
    plt.plot(alpha_values, markov_bounds, linestyle='-', color='r', label="Markov's Bound")
    plt.plot(alpha_values, chebyshev_bounds, linestyle='-', color='g', label="Chebyshev's Bound")
    plt.plot(alpha_values, hoeffding_bounds, linestyle='-', color='purple', label="Hoeffding's Bound")

    # Plot the empirical frequencies as vertical bars
    alpha_frequencies = [np.count_nonzero(sample_mean >= alpha) / num_repetitions for alpha in alpha_values]
    plt.vlines(alpha_values, 0, alpha_frequencies, color='b', linewidth=4, label='Empirical Frequency')

    plt.xlabel('Alpha')
    plt.ylabel('Frequency/Probability')
    plt.title('Empirical Frequency and Bounds')
    plt.legend()
    plt.grid(False)
    plt.ylim((0.0, 1.1))
    plt.xlim((alpha_start - 0.02, alpha_end + 0.02))
    plt.xticks([i for i in np.arange(alpha_start, alpha_end + 0.05, 0.05)])

    plt.show()

coin_bias_1 = 0.5
alpha_start_1 = 0.5
alpha_end_1 = 1.0
alpha_step_1 = 0.05

calculate_bounds_and_plot(coin_bias_1, alpha_start_1, alpha_end_1, alpha_step_1)

coin_bias_2 = 0.1
alpha_start_2 = 0.1
alpha_end_2 = 1.0
alpha_step_2 = 0.05

calculate_bounds_and_plot(coin_bias_2, alpha_start_2, alpha_end_2, alpha_step_2)


# Task 1A probability calculations
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

# Task 1B probability calculations
####################################################################################################

# Parameters for the binomial distribution
n2 = 20
p2 = 0.1

# Define the alpha values
alpha_1_2 = 1
alpha_0_95 = 0.95

# Calculate the probabilities using the binomial CDF
probability_alpha_1_2 = 1 - binom.cdf(n2 * alpha_1_2 - 1, n2, p2)
probability_alpha_0_95_2 = 1 - binom.cdf(n2 * alpha_0_95 - 1, n2, p2)

print(f'P(1/20 * sum(X_i) >= {alpha_1_2}) = {probability_alpha_1_2:.20f}')
print(f'P(1/20 * sum(X_i) >= {alpha_0_95}) = {probability_alpha_0_95_2:.20f}')

####################################################################################################