# EXACT MATCH TO COLAB

import statsmodels.formula.api as sm

# Apply exact transformation
df['tr_score'] = 1 - (1 / df[response])

# Fit model exactly like Colab
model = sm.ols(
    formula='tr_score ~ wrkyrs + C(desgn, Treatment(reference="staff")) + C(priorQI, Treatment(reference="no"))',
    data=df
).fit()

st.text(model.summary())

# Match printed values exactly
st.write("MSE:", np.sqrt(model.mse_resid))
st.write("MLE:", np.sqrt(model.ssr / model.nobs))
