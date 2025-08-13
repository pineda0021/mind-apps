import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
from fractions import Fraction


def print_footer():
    print("\nProfessor: Edward Pineda-Castro")
    print("Department of Mathematics")
    print("Los Angeles City College")


def parse_input(input_string, allow_fraction=False):
    try:
        if allow_fraction:
            return np.array([float(Fraction(x.strip())) for x in input_string.split(',')])
        else:
            return np.array([float(x.strip()) for x in input_string.split(',')])
    except ValueError:
        print("❌ Error: Please enter only numbers or fractions separated by commas.")
        return None


def validate_probabilities(px):
    return np.isclose(np.sum(px), 1.0)


def display_discrete_summary(X, P):
    mean = np.round(np.sum(X * P), 4)
    variance = np.round(np.sum(((X - mean) ** 2) * P), 4)
    std_dev = np.round(np.sqrt(variance), 4)
    cumulative = np.round(np.cumsum(P), 4)

    print("\n📊 Discrete Probability Distribution Summary")
    print(f"{'X':<10}{'P(X)':<15}{'Cumulative P(X ≤ x)'}")
    print("-" * 40)
    for x, p, c in zip(X, P, cumulative):
        print(f"{x:<10}{p:<15}{c}")

    print("\n🧮 Descriptive Statistics:")
    print(f"{'Mean (μ)':<22}: {mean}")
    print(f"{'Variance (σ²)':<22}: {variance}")
    print(f"{'Std Dev (σ)':<22}: {std_dev}")


def display_probability_bar(X, P):
    fig = go.Figure(data=[go.Bar(x=X, y=P, marker_color='green')])
    fig.update_layout(
        title="Probability Distribution of X",
        xaxis_title="X",
        yaxis_title="P(X)",
        template="simple_white"
    )
    fig.show()


def general_discrete_distribution():
    while True:
        print("\n--- General Discrete Probability Distribution ---")
        print("Enter 'exit' at any time to quit.")

        x_input = input("Enter values of random variable X (comma-separated): ").strip()
        if x_input.lower() == 'exit':
            break

        p_input = input("Enter corresponding probabilities P(X) (comma-separated, fractions allowed): ").strip()
        if p_input.lower() == 'exit':
            break

        X = parse_input(x_input)
        P = parse_input(p_input, allow_fraction=True)

        if X is None or P is None:
            continue
        if len(X) != len(P):
            print("❌ Error: X and P(X) must be of the same length.")
            continue
        if not validate_probabilities(P):
            print(f"❌ Error: Probabilities must sum to 1. Your sum was {np.sum(P):.4f}")
            continue

        display_discrete_summary(X, P)
        display_probability_bar(X, P)

        again = input("\nWould you like to analyze another discrete distribution? (yes/no): ").strip().lower()
        if again != 'yes':
            break


def parse_fraction(p_str):
    """Parse string to float from decimal or fraction (e.g., '1/2')."""
    try:
        return float(Fraction(p_str.strip()))
    except ValueError:
        raise ValueError("Invalid probability format. Please enter a decimal (e.g., 0.5) or a fraction (e.g., 1/2).")


def display_binomial_table(n, p):
    print("\n📋 Binomial Probability Distribution Table:")
    print(f"{'x':>3} {'P(X = x)':>12}")
    print("-" * 18)
    for x in range(n + 1):
        prob = binom.pmf(x, n, p)
        print(f"{x:>3} {prob:>12.5f}")


def display_binomial_plot(n, p):
    x = np.arange(0, n + 1)
    y = binom.pmf(x, n, p)

    plt.figure(figsize=(10, 5))
    plt.bar(x, y, color='skyblue', edgecolor='black')
    plt.title(f'Binomial Distribution (n={n}, p={p})')
    plt.xlabel('Number of Successes')
    plt.ylabel('Probability')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()


def binomial_distribution():
    print("\n🎲 Binomial Probability Calculator")

    while True:
        try:
            n = int(input("🔢 Enter the number of trials (n): "))
            p_input = input("🎯 Enter the probability of success (p — decimals like 0.4 or fractions like 1/2): ")
            p = parse_fraction(p_input)
            if not (0 <= p <= 1):
                raise ValueError("❌ Probability must be between 0 and 1.")
            break
        except ValueError as ve:
            print(ve)

    while True:
        print("\nChoose the type of probability calculation:")
        print("1. Exactly x successes")
        print("2. At most x successes")
        print("3. At least x successes")
        print("4. Between x and y successes")
        print("5. More than x successes")
        print("6. Fewer than x successes")
        print("7. View Probability Table and Graph")
        print("8. Exit")

        try:
            choice = int(input("Enter your choice (1–8): "))
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 8.")
            continue

        if choice == 1:
            x = int(input("Enter the number of successes (x): "))
            probability = binom.pmf(x, n, p)
            print(f"📌 P(X = {x}) = {probability:.5f}")
        elif choice == 2:
            x = int(input("Enter the number of successes (x): "))
            probability = binom.cdf(x, n, p)
            print(f"📌 P(X ≤ {x}) = {probability:.5f}")
        elif choice == 3:
            x = int(input("Enter the number of successes (x): "))
            probability = 1 - binom.cdf(x - 1, n, p)
            print(f"📌 P(X ≥ {x}) = {probability:.5f}")
        elif choice == 4:
            a = int(input("Enter the lower bound (a): "))
            b = int(input("Enter the upper bound (b): "))
            probability = binom.cdf(b, n, p) - binom.cdf(a - 1, n, p)
            print(f"📌 P({a} ≤ X ≤ {b}) = {probability:.5f}")
        elif choice == 5:
            x = int(input("Enter the number of successes (x): "))
            probability = 1 - binom.cdf(x, n, p)
            print(f"📌 P(X > {x}) = {probability:.5f}")
        elif choice == 6:
            x = int(input("Enter the number of successes (x): "))
            probability = binom.cdf(x - 1, n, p)
            print(f"📌 P(X < {x}) = {probability:.5f}")
        elif choice == 7:
            display_binomial_table(n, p)
            display_binomial_plot(n, p)
        elif choice == 8:
            print("Exiting binomial calculator.")
            print_footer()
            break
        else:
            print("❌ Invalid choice. Please enter a number between 1 and 8.")


def display_poisson_table(lam, max_k):
    print("\n📋 Poisson Probability Distribution Table:")
    print(f"{'k':>3} {'P(X = k)':>12}")
    print("-" * 18)
    for k in range(max_k + 1):
        prob = poisson.pmf(k, lam)
        print(f"{k:>3} {prob:>12.5f}")


def display_poisson_plot(lam, max_k):
    x = np.arange(0, max_k + 1)
    y = poisson.pmf(x, lam)

    plt.figure(figsize=(10, 5))
    plt.bar(x, y, color='coral', edgecolor='black')
    plt.title(f'Poisson Distribution (λ={lam})')
    plt.xlabel('Number of Events (k)')
    plt.ylabel('Probability')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()


def poisson_distribution():
    print("\n🐝 Poisson Probability Calculator")

    while True:
        try:
            lam = float(input("Enter the average rate (λ > 0): "))
            if lam <= 0:
                raise ValueError("❌ λ must be greater than 0.")
            break
        except ValueError as ve:
            print(ve)

    max_k = int(input("Enter the maximum k value to display (e.g., 20): "))

    while True:
        print("\nChoose the type of probability calculation:")
        print("1. Exactly k events")
        print("2. At most k events")
        print("3. At least k events")
        print("4. Between k1 and k2 events")
        print("5. More than k events")
        print("6. Fewer than k events")
        print("7. View Probability Table and Graph")
        print("8. Exit")

        try:
            choice = int(input("Enter your choice (1–8): "))
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 8.")
            continue

        if choice == 1:
            k = int(input("Enter the number of events (k): "))
            probability = poisson.pmf(k, lam)
            print(f"📌 P(X = {k}) = {probability:.5f}")
        elif choice == 2:
            k = int(input("Enter the number of events (k): "))
            probability = poisson.cdf(k, lam)
            print(f"📌 P(X ≤ {k}) = {probability:.5f}")
        elif choice == 3:
            k = int(input("Enter the number of events (k): "))
            probability = 1 - poisson.cdf(k - 1, lam)
            print(f"📌 P(X ≥ {k}) = {probability:.5f}")
        elif choice == 4:
            k1 = int(input("Enter the lower bound (k1): "))
            k2 = int(input("Enter the upper bound (k2): "))
            probability = poisson.cdf(k2, lam) - poisson.cdf(k1 - 1, lam)
            print(f"📌 P({k1} ≤ X ≤ {k2}) = {probability:.5f}")
        elif choice == 5:
            k = int(input("Enter the number of events (k): "))
            probability = 1 - poisson.cdf(k, lam)
            print(f"📌 P(X > {k}) = {probability:.5f}")
        elif choice == 6:
            k = int(input("Enter the number of events (k): "))
            probability = poisson.cdf(k - 1, lam)
            print(f"📌 P(X < {k}) = {probability:.5f}")
        elif choice == 7:
            display_poisson_table(lam, max_k)
            display_poisson_plot(lam, max_k)
        elif choice == 8:
            print("Exiting Poisson calculator.")
            print_footer()
            break
        else:
            print("❌ Invalid choice. Please enter a number between 1 and 8.")


def run():
    print("\n=== Discrete Distributions Tool ===")
    while True:
        print("\nChoose a distribution to analyze:")
        print("1. General Discrete Distribution (User inputs values and probabilities)")
        print("2. Binomial Distribution")
        print("3. Poisson Distribution")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ").strip()
        if choice == '1':
            general_discrete_distribution()
        elif choice == '2':
            binomial_distribution()
        elif choice == '3':
            poisson_distribution()
        elif choice == '4':
            print("Exiting Discrete Distributions Tool.")
            print_footer()
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    run()
