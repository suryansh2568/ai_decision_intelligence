import pandas as pd
import numpy as np
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, '..', 'data', 'online_retail.csv')
output_path = os.path.join(script_dir, '..', 'data', 'retail_daily_aggregated.csv')

print(f"Attempting to load: {file_path}")

try:
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print("File loaded successfully!")
except FileNotFoundError:
    print(f"ERROR: Could not find file at {file_path}")
    print("Please ensure 'online_retail.csv' is inside the 'data' folder.")
    exit()

# Data Cleaning
df_clean = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()

# Fix Dates
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
df_clean['Date'] = df_clean['InvoiceDate'].dt.date

# Select Top Products
df_clean['Revenue'] = df_clean['Quantity'] * df_clean['UnitPrice']
top_products = df_clean.groupby('StockCode')['Revenue'].sum().sort_values(ascending=False).head(10).index.tolist()

print(f"Modeling Top 10 Products: {top_products}")
df_top = df_clean[df_clean['StockCode'].isin(top_products)].copy()

# Grouping
daily_data = df_top.groupby(['Date', 'StockCode']).agg({
    'Quantity': 'sum',       # Total sold that day
    'UnitPrice': 'mean',     # Average price that day
    'Description': 'first'   # Keep the name
}).reset_index()

daily_data.columns = ['Date', 'SKU_ID', 'Quantity_Sold', 'Price', 'Description']

# Feature Engineering
daily_data['Date'] = pd.to_datetime(daily_data['Date'])
daily_data['DayOfWeek'] = daily_data['Date'].dt.dayofweek
daily_data['Is_Weekend'] = daily_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Simulate Competitor Price
np.random.seed(42)
daily_data['Competitor_Price'] = daily_data['Price'] * np.random.uniform(0.9, 1.1, size=len(daily_data))

# Save the prepared data to the 'data' folder
daily_data.to_csv(output_path, index=False)

print(f"Transformation Complete. Saved to: {output_path}")
print(daily_data.head())