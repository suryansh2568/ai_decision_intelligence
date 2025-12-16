import pandas as pd
import numpy as np
import joblib
import os

# Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', 'retail_daily_aggregated.csv')
model_path = os.path.join(script_dir, '..', 'data', 'demand_model_v2.pkl')

print("Loading model and data...")
artifacts = joblib.load(model_path)
model = artifacts['model']
feature_names = artifacts['features']

df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])

# Optimization Function
def optimize_price(sku_id, target_date):
    # A. Getting the Base Data for this Product/Date
    base_row = df[(df['SKU_ID'] == sku_id) & (df['Date'] == target_date)].copy()
    
    if base_row.empty:
        print(f"No data found for {sku_id} on {target_date}")
        return None

    # B. Defining Price Range to Test
    current_price = base_row['Price'].values[0]
    min_price = current_price * 0.8  # -20%
    max_price = current_price * 1.5  # +50%
    
    # Testing 20 different price points in this range
    test_prices = np.linspace(min_price, max_price, 20)
    
    results = []
    
    print(f"\n--- Optimizing {sku_id} for {target_date.date()} ---")
    print(f"Current Price: ${current_price:.2f}")

    # C. Simulation Loop
    for test_price in test_prices:
        simulated_row = base_row.copy()
        
        # 1. Updating Price
        simulated_row['Price'] = test_price
        
        # 2. Update Competitor Price (Assuming competitor price is static)
        
        # 3. Preparing Input for Model
        simulated_row = pd.get_dummies(simulated_row, columns=['SKU_ID'], prefix='SKU')
        
        # Adding missing columns with 0
        for col in feature_names:
            if col not in simulated_row.columns:
                simulated_row[col] = 0
                
        # Selecting features in correct order
        X_input = simulated_row[feature_names]
        
        # 4. Predicting Demand
        log_pred = model.predict(X_input)[0]
        pred_quantity = np.expm1(log_pred)
        
        # 5. Calculating Revenue
        pred_quantity = max(0, pred_quantity)
        projected_revenue = test_price * pred_quantity
        
        results.append({
            'Price': test_price,
            'Predicted_Demand': pred_quantity,
            'Revenue': projected_revenue
        })

    # D. Finding Winner
    results_df = pd.DataFrame(results)
    best_scenario = results_df.loc[results_df['Revenue'].idxmax()]
    
    print("\n🏆 OPTIMIZATION RESULT 🏆")
    print(f"Optimal Price: ${best_scenario['Price']:.2f}")
    print(f"Predicted Sales: {best_scenario['Predicted_Demand']:.1f} units")
    print(f"Projected Revenue: ${best_scenario['Revenue']:.2f}")
    
    revenue_lift = (best_scenario['Revenue'] - (current_price * base_row['Quantity_Sold'].values[0]))
    print(f"Revenue Impact vs Actual: ${revenue_lift:.2f}")
    
    return results_df

# Running it for a Top Product
sample_date = pd.Timestamp('2011-11-08')

# Picking the most popular product
top_sku = df['SKU_ID'].value_counts().idxmax()

optimization_results = optimize_price(top_sku, sample_date)