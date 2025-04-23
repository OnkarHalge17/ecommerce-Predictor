import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv('sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month

# Encode categories
df['Product_Category'] = df['Product_Category'].astype('category').cat.codes

# Features and Target
X = df[['Product_ID', 'Product_Category', 'Quantity_Sold', 'Promotions', 'Day', 'Month']]
y = df['Total_Sales']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
