
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Epsilon–Delta Visualizer", layout="centered")

st.title("🧠 Epsilon–Delta Limit Visualizer")
st.markdown("### Explore how delta (𝛿) and epsilon (𝜖) work in the definition of a limit.")

# Select function
function_choice = st.selectbox("Select a function", [
    "f(x) = x²",
    "f(x) = 1/x",
    "f(x) = sin(x)",
    "f(x) = e^x",
    "f(x) = ln(x)",
    "f(x) = sqrt(x)"
])

# Define functions
def f(x):
    if function_choice == "f(x) = x²":
        return x**2
    elif function_choice == "f(x) = 1/x":
        return 1/x
    elif function_choice == "f(x) = sin(x)":
        return np.sin(x)
    elif function_choice == "f(x) = e^x":
        return np.exp(x)
    elif function_choice == "f(x) = ln(x)":
        return np.log(x)
    elif function_choice == "f(x) = sqrt(x)":
        return np.sqrt(x)

# Default 'a' values based on function
default_a_values = {
    "f(x) = x²": 2,
    "f(x) = 1/x": 1,
    "f(x) = sin(x)": np.pi,
    "f(x) = e^x": 1,
    "f(x) = ln(x)": 1,
    "f(x) = sqrt(x)": 4,
}
a = st.number_input("Choose a value of a (where x → a)", value=float(default_a_values[function_choice]), format="%.2f")
L = f(a)

# Sliders for epsilon and delta
epsilon = st.slider("ε (epsilon)", min_value=0.01, max_value=2.0, value=0.5, step=0.01)
delta = st.slider("δ (delta)", min_value=0.01, max_value=2.0, value=0.5, step=0.01)

# Visualization
x = np.linspace(a - 2, a + 2, 1000)
with np.errstate(divide='ignore', invalid='ignore'):
    y = f(x)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, label=function_choice)
ax.axvline(a, color='black', linestyle='--', label=r'$x = a$')
ax.axhline(L, color='green', linestyle='--', label=r'$L = f(a)$')

# Epsilon band
ax.axhline(L + epsilon, color='red', linestyle='--', label=r'$\epsilon$ band')
ax.axhline(L - epsilon, color='red', linestyle='--')

# Delta band
ax.axvline(a + delta, color='blue', linestyle='--', label=r'$\delta$ band')
ax.axvline(a - delta, color='blue', linestyle='--')

# Valid x values
mask = (np.abs(x - a) < delta) & (x != a)
valid_x = x[mask]
valid_y = f(valid_x)
ax.plot(valid_x, valid_y, 'o', color='purple', label=r'$|f(x) - L| < \epsilon$')

ax.set_title(f"Epsilon–Delta Visualization for {function_choice} as x → {a}")
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_ylim(np.nanmin(y[np.isfinite(y)]) - 1, np.nanmax(y[np.isfinite(y)]) + 1)
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Explanation
st.markdown("---")
st.markdown("### 📘 Limit Definition Reminder:")
st.latex(r"\lim_{x 	o a} f(x) = L")
st.latex(fr"\lim_{{x 	o {a}}} f(x) = {L:.4f}")
st.markdown("if for every")
st.latex(r"arepsilon > 0")
st.markdown("there exists a")
st.latex(r"\delta > 0")
st.markdown("such that whenever")
st.latex(fr"0 < |x - {a}| < \delta")
st.markdown("we have")
st.latex(fr"|f(x) - {L:.4f}| < arepsilon")

# Footer
st.markdown("---")
st.markdown("🔎 This tool helps you see how choosing a value of \( \delta \) affects which \( x \)-values keep \( f(x) \) within an \( \varepsilon \)-band around \( L \).")
st.markdown("🧡 Created by **Professor Edward Pineda-Castro**, Los Angeles City College — built with the students in **MIND**.")
