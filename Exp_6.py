# Importing necessary libraries
import numpy as np
from scipy import stats
# Generating sample data
np.random.seed(42)  # For reproducibility
sample1 = np.random.normal(loc=10, scale=5, size=100)  # Sample data from population 1
sample2 = np.random.normal(loc=12, scale=5, size=100)  # Sample data from population 2
# Performing a two-sample t-test
t_statistic, p_value = stats.ttest_ind(sample1, sample2)
# Displaying the results
print("T-statistic:", t_statistic)
print("P-value:", p_value)
# Interpreting the results
alpha = 0.05  # Significance level
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the populations.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the populations.")
