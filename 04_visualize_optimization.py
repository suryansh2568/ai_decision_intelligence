import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# 1. Generating Synthetic Data
np.random.seed(42) # Ensures the graph looks the same every time

# Creating simulated transaction data
n_samples = 1000
simulated_prices = np.random.uniform(5, 20, n_samples)

# Demand Function: Quantity = 100 - 4 * Price + Noise
simulated_quantities = 100 - 4 * simulated_prices + np.random.normal(0, 5, n_samples)

# Clean up: Quantity can't be negative
simulated_quantities = np.maximum(simulated_quantities, 0)

# Creating a DataFrame
df_synthetic = pd.DataFrame({'UnitPrice': simulated_prices, 'Quantity': simulated_quantities})
print("Synthetic Data Generated (Simulating a Price-Elastic Product)")


# 2. Training the Model (Polynomial Degree 2)
# Aggregating to get the demand curve
demand_curve = df_synthetic.groupby(pd.cut(df_synthetic['UnitPrice'], bins=20)).agg({
    'UnitPrice': 'mean',
    'Quantity': 'sum'
}).dropna()

X = demand_curve['UnitPrice'].values.reshape(-1, 1)
y = demand_curve['Quantity'].values

# Fitting Polynomial (Quadratic) Model: Revenue = P * Q(P)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)


# 3. Generating Optimization Curve
# Testing prices from $5 to $20
price_range = np.linspace(5, 20, 100).reshape(-1, 1)
price_range_poly = poly.transform(price_range)

# Predicting Quantity
predicted_qty = model.predict(price_range_poly)
predicted_qty = np.maximum(predicted_qty, 0) # Clip negative demand

# Calculating Revenue
predicted_revenue = price_range.flatten() * predicted_qty

# Finding Optimal Price
best_idx = np.argmax(predicted_revenue)
opt_price = price_range.flatten()[best_idx]
max_rev = predicted_revenue[best_idx]


# 4. Plotting the Curve
plt.figure(figsize=(10, 6))

plt.plot(price_range, predicted_revenue, color='green', linewidth=3, label='Projected Revenue')
plt.axvline(opt_price, color='red', linestyle='--', label=f'Optimal Price: ${opt_price:.2f}')
plt.scatter(opt_price, max_rev, color='red', s=100, zorder=5)

plt.title('Theoretical Price Optimization (Synthetic Data)', fontsize=14)
plt.xlabel('Price ($)', fontsize=12)
plt.ylabel('Projected Revenue ($)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()