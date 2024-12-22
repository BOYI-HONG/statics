import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, probplot


np.random.seed(42)  #set a random generator 
mean_exponential = 10 # mean for exponential
n_samples = 50 # sample size
original_sample = np.random.exponential(scale=mean_exponential, size=n_samples) 
# generate a sample from exponential distribution
# pdf of exponential distribution
x = np.linspace(0, np.max(original_sample), 500)
theoretical_pdf = expon.pdf(x, scale=mean_exponential)

plt.hist(original_sample, bins=15, density=True, alpha=0.7, color="blue", label="Original Sample")
plt.plot(x, theoretical_pdf, color="red", label="Theoretical Exponential Distribution")
plt.title("Original Sample vs Theoretical Exponential Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.savefig('Original_Sample_vs_Theoretical_Exponential_Distribution.png', transparent=True)
# plt.show()

# Bootstrap method 
bootstrap_means = []
n_bootstrap_samples = 30
n_iterations = 100

for _ in range(n_iterations):
    bootstrap_sample = np.random.choice(original_sample, size=n_bootstrap_samples, replace=True)
    bootstrap_means.append(np.mean(bootstrap_sample))

# 1. mean of bootstrap sample distribution
bootstrap_means = np.array(bootstrap_means)
percentile_5 = np.percentile(bootstrap_means, 5)
percentile_95 = np.percentile(bootstrap_means, 95)
print("5th Percentile:", percentile_5)
print("95th Percentile:", percentile_95)

plt.hist(bootstrap_means, bins=20, density=True, alpha=0.7, color="blue", label="Bootstrap Means")
plt.axvline(percentile_5, color="red", linestyle="--", label="5th Percentile")
plt.axvline(percentile_95, color="green", linestyle="--", label="95th Percentile")
plt.title("Probability Distribution of Bootstrap Sample Means")
plt.xlabel("Sample Mean")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.savefig('Probability_Distribution_of_Bootstrap_Sample_Means.png', transparent=True)
plt.close()


# 2 Normal Q-Q plot of bootstrap sample means
res_norm = probplot(bootstrap_means, dist="norm", plot=plt)

theoretical_quantiles_norm = res_norm[0][0]  
observed_quantiles_norm = res_norm[0][1]     

# Normal Q-Q plot R² value 
slope_norm, intercept_norm = np.polyfit(theoretical_quantiles_norm, observed_quantiles_norm, 1)
fitted_quantiles_norm = slope_norm * theoretical_quantiles_norm + intercept_norm
residuals_norm = observed_quantiles_norm - fitted_quantiles_norm
ss_res_norm = np.sum(residuals_norm ** 2)  # 殘差平方和
ss_tot_norm = np.sum((observed_quantiles_norm - np.mean(observed_quantiles_norm)) ** 2)  # 總平方和
r_squared_norm = 1 - (ss_res_norm / ss_tot_norm)

print("R² value from Normal Q-Q plot:", r_squared_norm)

plt.scatter(res_norm[0][0], res_norm[0][1], color='blue', label="Data Points")


plt.plot(res_norm[0][0], res_norm[1][0] * res_norm[0][0] + res_norm[1][1], color='red', label="Fitting Line")
plt.legend()
plt.xlabel("Theoretical Quantiles")
plt.grid(True)
plt.title("Normal Probability Plot of Bootstrap Sample Means")
plt.savefig('Normal_Probability_Plot_of_Bootstrap_Sample_Means.png', transparent=True)

plt.close()

# 3 exponential Q-Q plot of original sample
# R² value from Exponential Q-Q plot
res_exp = probplot(original_sample, dist=expon, plot=plt)
slope, intercept = res_exp[1][0], res_exp[1][1]

theoretical_quantiles_exp = res_exp[0][0]  
observed_quantiles_exp = res_exp[0][1]     

slope_exp, intercept_exp = np.polyfit(theoretical_quantiles_exp, observed_quantiles_exp, 1)
fitted_quantiles_exp = slope_exp * theoretical_quantiles_exp + intercept_exp
residuals_exp = observed_quantiles_exp - fitted_quantiles_exp
ss_res_exp = np.sum(residuals_exp ** 2)  # 殘差平方和
ss_tot_exp = np.sum((observed_quantiles_exp - np.mean(observed_quantiles_exp)) ** 2)  # 總平方和
r_squared_exp = 1 - (ss_res_exp / ss_tot_exp)

print("R² value from Exponential Q-Q plot:", r_squared_exp)

plt.scatter(res_exp[0][0], res_exp[0][1], color='blue', label="Data Points")
plt.plot(res_exp[0][0], res_exp[1][0] * res_exp[0][0] + res_exp[1][1], color='red', label="Fitting Line")
plt.legend()
plt.xlabel("Theoretical Quantiles")
plt.title("Exponential Probability Plot of Original Sample")
plt.grid(True)
plt.savefig('Exponential_Probability_Plot_of_Original_Sample.png', transparent=True)
plt.close()
'''
# true mean of original sample
estimated_mean = np.mean(original_sample)
print("Estimated Mean from Original Sample:", estimated_mean)
'''
