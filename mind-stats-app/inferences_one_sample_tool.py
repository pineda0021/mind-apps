# inferences_one_sample_cli.py
import math
import numpy as np
from scipy.stats import norm, t, chi2, binom

def proportion_large(x, n, p0, alpha, tails, dp):
    p_hat = x / n
    se = math.sqrt(p0 * (1 - p0) / n)
    z = (p_hat - p0) / se

    if tails == "two":
        crit_low, crit_high = -norm.ppf(1 - alpha/2), norm.ppf(1 - alpha/2)
        p_val = 2 * (1 - norm.cdf(abs(z)))
    elif tails == "left":
        crit_low, crit_high = norm.ppf(alpha), None
        p_val = norm.cdf(z)
    else:
        crit_low, crit_high = None, norm.ppf(1 - alpha)
        p_val = 1 - norm.cdf(z)

    conclusion = "Reject the null hypothesis" if (
        (tails == "two" and abs(z) > crit_high) or
        (tails == "left" and z < crit_low) or
        (tails == "right" and z > crit_high)
    ) else "Do not reject the null hypothesis"

    print_crit(crit_low, crit_high, dp)
    print(f"Test statistic: {z:.{dp}f}")
    print(f"P-value: {p_val:.{dp}f}")
    print(f"Conclusion: {conclusion}")

def proportion_small(x, n, p0, alpha, tails, dp):
    if tails == "two":
        p_val = 2 * min(binom.cdf(x, n, p0), 1 - binom.cdf(x - 1, n, p0))
    elif tails == "left":
        p_val = binom.cdf(x, n, p0)
    else:
        p_val = 1 - binom.cdf(x - 1, n, p0)

    conclusion = "Reject the null hypothesis" if p_val < alpha else "Do not reject the null hypothesis"

    print(f"P-value: {p_val:.{dp}f}")
    print(f"Conclusion: {conclusion}")

def t_test_summary(mean, s, n, mu0, alpha, tails, dp):
    df = n - 1
    se = s / math.sqrt(n)
    t_stat = (mean - mu0) / se

    if tails == "two":
        crit_low, crit_high = -t.ppf(1 - alpha/2, df), t.ppf(1 - alpha/2, df)
        p_val = 2 * (1 - t.cdf(abs(t_stat), df))
    elif tails == "left":
        crit_low, crit_high = t.ppf(alpha, df), None
        p_val = t.cdf(t_stat, df)
    else:
        crit_low, crit_high = None, t.ppf(1 - alpha, df)
        p_val = 1 - t.cdf(t_stat, df)

    conclusion = "Reject the null hypothesis" if (
        (tails == "two" and abs(t_stat) > crit_high) or
        (tails == "left" and t_stat < crit_low) or
        (tails == "right" and t_stat > crit_high)
    ) else "Do not reject the null hypothesis"

    print_crit(crit_low, crit_high, dp)
    print(f"Test statistic: {t_stat:.{dp}f}")
    print(f"P-value: {p_val:.{dp}f}")
    print(f"Conclusion: {conclusion}")

def chi_sq_summary(s, n, sigma0, alpha, tails, dp):
    df = n - 1
    chi_stat = (df * s**2) / sigma0**2

    if tails == "two":
        crit_low, crit_high = chi2.ppf(alpha/2, df), chi2.ppf(1 - alpha/2, df)
        p_val = 2 * min(chi2.cdf(chi_stat, df), 1 - chi2.cdf(chi_stat, df))
    elif tails == "left":
        crit_low, crit_high = chi2.ppf(alpha, df), None
        p_val = chi2.cdf(chi_stat, df)
    else:
        crit_low, crit_high = None, chi2.ppf(1 - alpha, df)
        p_val = 1 - chi2.cdf(chi_stat, df)

    conclusion = "Reject the null hypothesis" if (
        (tails == "two" and (chi_stat < crit_low or chi_stat > crit_high)) or
        (tails == "left" and chi_stat < crit_low) or
        (tails == "right" and chi_stat > crit_high)
    ) else "Do not reject the null hypothesis"

    print_crit(crit_low, crit_high, dp)
    print(f"Test statistic: {chi_stat:.{dp}f}")
    print(f"P-value: {p_val:.{dp}f}")
    print(f"Conclusion: {conclusion}")

def print_crit(low, high, dp):
    if low is not None and high is not None:
        print(f"Critical Values: {low:.{dp}f}, {high:.{dp}f}")
    elif low is not None:
        print(f"Critical Value: {low:.{dp}f}")
    elif high is not None:
        print(f"Critical Value: {high:.{dp}f}")

# ===== Main Program =====
print("Inferences on One Sample")
print("1. Proportion Test (Large Sample)")
print("2. Proportion Test (Small Sample - Binomial)")
print("3. t-Test for Population Mean (Summary Stats)")
print("4. t-Test for Population Mean (Raw Data)")
print("5. Chi-Squared Test for Standard Deviation (Summary Stats)")
print("6. Chi-Squared Test for Standard Deviation (Raw Data)")

choice = int(input("Enter your choice (1-6): "))
alpha = float(input("Enter significance level (e.g., 0.05): "))
tails = input("Enter tails ('left', 'right', or 'two'): ").strip().lower()
dp = int(input("Decimal places for rounding: "))

if choice == 1:
    x = int(input("Enter number of successes: "))
    n = int(input("Enter sample size: "))
    p0 = float(input("Enter null proportion (p0): "))
    proportion_large(x, n, p0, alpha, tails, dp)

elif choice == 2:
    x = int(input("Enter number of successes: "))
    n = int(input("Enter sample size: "))
    p0 = float(input("Enter null proportion (p0): "))
    proportion_small(x, n, p0, alpha, tails, dp)

elif choice == 3:
    mean = float(input("Enter sample mean: "))
    s = float(input("Enter sample standard deviation: "))
    n = int(input("Enter sample size: "))
    mu0 = float(input("Enter null hypothesis mean (μ0): "))
    t_test_summary(mean, s, n, mu0, alpha, tails, dp)

elif choice == 4:
    raw = input("Enter raw data separated by commas: ")
    data = np.array(list(map(float, raw.split(","))))
    mu0 = float(input("Enter null hypothesis mean (μ0): "))
    mean, s, n = np.mean(data), np.std(data, ddof=1), len(data)
    t_test_summary(mean, s, n, mu0, alpha, tails, dp)

elif choice == 5:
    s = float(input("Enter sample standard deviation: "))
    n = int(input("Enter sample size: "))
    sigma0 = float(input("Enter population standard deviation (σ0): "))
    chi_sq_summary(s, n, sigma0, alpha, tails, dp)

elif choice == 6:
    raw = input("Enter raw data separated by commas: ")
    data = np.array(list(map(float, raw.split(","))))
    sigma0 = float(input("Enter population standard deviation (σ0): "))
    s, n = np.std(data, ddof=1), len(data)
    chi_sq_summary(s, n, sigma0, alpha, tails, dp)


