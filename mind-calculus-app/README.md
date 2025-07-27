# 🧠 MIND: Calculus Visualizer Suite

Created by **Professor Edward Pineda-Castro** at Los Angeles City College, this interactive Streamlit app helps students explore core calculus concepts using visuals, animations, and symbolic math.

## 🚀 Tools Included

1. **Limits Visualizer**
   - Explore removable and non-removable discontinuities
   - See animations of functions approaching a limit
   - Tabular values with undefined points
   - ✅ *Challenge*: Determine the limit as \( x \to c \) from a table and animation.

2. **Derivative Visualizer**
   - Plot a function and its derivative
   - See symbolic differentiation (via SymPy)
   - ✅ *Challenge*: Match a function to its derivative graph or compute \( f'(x) \) at a point.

3. **Riemann Sum Explorer**
   - Compare Left, Right, Midpoint, Trapezoidal, and Upper/Lower Sums
   - Overlay rectangles on graphs
   - Compute absolute and relative error
   - ✅ *Challenge*: Estimate area using Midpoint sum and compare to the exact integral.

4. **Antiderivative (Integral) Visualizer**
   - See the function and its integral side-by-side
   - Symbolic antiderivatives displayed with +C
   - ✅ *Challenge*: Identify antiderivatives from multiple choices.

5. **Solid of Revolution Tool**
   - Compute volume using Disk/Washer or Shell method
   - 3D plotly visualization for revolution around the x or y-axis
   - Step-by-step integral setup and solution in exact form
   - ✅ *Challenge*: Match volume expressions to the correct setup for a region.

## 🏁 Getting Started

### 🔧 Requirements
Make sure to install the following Python packages:

```
streamlit
numpy
matplotlib
sympy
plotly
```

### 📁 File Structure
```
mind-calculus-app/
├── main.py
├── limits_tool.py
├── derivative_tool.py
├── riemann_tool.py
├── antiderivative_tool.py
├── solid_volume_tool.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

Your app will be live at:
```
https://your-app-name.streamlit.app
```

## 🎓 Educational Value
Each module includes:
- 💡 Concept overview
- 📘 Challenge problems with randomized input
- ✏️ Symbolic solutions
- 📊 Visual exploration

Students can learn by adjusting parameters, testing examples, and verifying their solutions.
---
Made with ❤️ for Calculus students everywhere by **Professor Edward Pineda-Castro**.
