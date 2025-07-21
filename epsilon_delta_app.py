
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("ε–δ Limit Visualization Tool")
st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND.")

# Define available functions
functions = {
    "f(x) = x²": lambda x: x**2,
    "f(x) = 1/x": lambda x: 1/x,
    "f(x) = sin(x)": lambda x: np.sin(x),
    "f(x) = e^x": lambda x: np.exp(x),
    "f(x) = ln(x)": lambda x: np.log(x)
}

func_name = st.selectbox("Choose a function f(x):", list(functions.keys()))
f = functions[func_name]

a = st.slider("Choose a value of a (approach point):", min_value=0.5, max_value=5.0, step=0.1, value=2.0)
epsilon = st.slider("Choose ε (epsilon):", min_value=0.01, max_value=2.0, step=0.01, value=0.5)
delta = st.slider("Choose δ (delta):", min_value=0.01, max_value=2.0, step=0.01, value=0.5)

L = f(a)
x = np.linspace(max(0.01, a - 2), a + 2, 1000)
y = f(x)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, label=f'{func_name}', alpha=0.8)
ax.axvline(a, color='black', linestyle='--', label='x = a')
ax.axhline(L, color='green', linestyle='--', label='L = f(a)')

# Epsilon band
ax.axhline(L + epsilon, color='red', linestyle='--', label='ε band')
ax.axhline(L - epsilon, color='red', linestyle='--')
# Delta band
ax.axvline(a + delta, color='blue', linestyle='--', label='δ band')
ax.axvline(a - delta, color='blue', linestyle='--')

x_valid = x[(np.abs(x - a) < delta) & (x != a)]
y_valid = f(x_valid)
ax.plot(x_valid, y_valid, 'o', color='purple', label='|f(x) - L| < ε')

ax.set_title(f"Epsilon–Delta Visualization for {func_name} as x → {a}")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Explanation
st.markdown("---")
st.markdown("### 📘 Limit Definition Reminder:")
st.latex(r"\lim_{x \to a} f(x) = L")
st.latex(fr"\lim_{{x \to {a}}} f(x) = {L:.4f}")

st.markdown("If for every:")
st.latex(r"\epsilon > 0")
st.markdown("there exists a:")
st.latex(r"\delta > 0")
st.markdown("such that whenever:")
st.latex(r"0 < |x - {a}| < \delta")
st.markdown("we have:")
st.latex(r"|f(x) - {L:.4f}| < \epsilon")

st.markdown("---")
st.markdown("🔎 This tool helps you see how choosing a value of δ affects which x-values keep f(x) within an ε-band around L.")
