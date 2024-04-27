from scipy.stats import norm
# Sample data
sample_mean = 25
population_mean = 20
population_std = 5
sample_size = 100
# Calculate the z-score
z_score = (sample_mean - population_mean) / (population_std / (sample_size ** 0.5))
# Calculate the p-value (two-tailed test)
p_value = 2 * (1 - norm.cdf(abs(z_score)))
print(f"Z-score: {z_score:.2f}")
print(f"P-value: {p_value:.4f}")
from scipy.stats import ttest_1samp
import numpy as np
# Sample data
sample_mean = 25
population_mean = 20
sample_std = 6
sample_size = 100
# Generate sample data
np.random.seed(42)  # for reproducibility
sample_data = np.random.normal(loc=sample_mean, scale=sample_std, size=sample_size)
# Perform the one-sample t-test
t_statistic, p_value = ttest_1samp(a=sample_data, popmean=population_mean)
print(f"T-statistic: {t_statistic:.2f}")
print(f"P-value: {p_value:.4f}")
import numpy as np
def z_test(sample_mean, population_mean, population_std, sample_size):
    # Calculate z-score
    z_score = (sample_mean - population_mean) / (population_std / (sample_size**0.5))
    # Calculate p-value (two-tailed test)
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    return z_score, p_value
# Sample data
sample_mean = 25
population_mean = 20
population_std = 5
sample_size = 100
# Perform z-test
z_score, p_value = z_test(sample_mean, population_mean, population_std, sample_size)
print(f"Z-score: {z_score:.2f}")
print(f"P-value: {p_value:.4f}")
import numpy as np
from scipy.stats import t
def t_test(sample_data, population_mean):
    sample_size = len(sample_data)
    sample_mean = np.mean(sample_data)
    sample_std = np.std(sample_data, ddof=1)  # ddof=1 for sample standard deviation
    # Calculate t-statistic
    t_statistic = (sample_mean - population_mean) / (sample_std / (sample_size**0.5))
    # Calculate p-value
    degrees_of_freedom = sample_size - 1
    p_value = 2 * (1 - t.cdf(abs(t_statistic), df=degrees_of_freedom))
    
    return t_statistic, p_value
# Sample data
sample_mean = 25
population_mean = 20
sample_size = 100
np.random.seed(42)  # for reproducibility
sample_data = np.random.normal(loc=sample_mean, scale=6, size=sample_size)
# Perform t-test
t_statistic, p_value = t_test(sample_data, population_mean)
print(f"T-statistic: {t_statistic:.2f}")
print(f"P-value: {p_value:.4f}")
