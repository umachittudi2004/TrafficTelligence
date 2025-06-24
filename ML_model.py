import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import pickle

# --- Load Dataset ---
data = pd.read_csv("traffic volume.csv")

# --- Fill Missing Values ---
'''data['temp'].fillna(data['temp'].mean(), inplace=True)
data['rain'].fillna(data['rain'].mean(), inplace=True)
data['snow'].fillna(data['snow'].mean(), inplace=True)
data['weather'] = data['weather'].fillna('Clouds')  # Best for just one column
'''
data.fillna({
    'temp': data['temp'].mean(),
    'rain': data['rain'].mean(),
    'snow': data['snow'].mean(),
    'weather': 'Clouds'
}, inplace=True)
data['holiday'] = data['holiday'].fillna('None')

# --- Split Date & Time ---
data[['day', 'month', 'year']] = data['date'].str.split('-', expand=True).astype(int)
data[['hours', 'minutes', 'seconds']] = data['Time'].str.split(':', expand=True).astype(int)
data.drop(columns=['date', 'Time'], inplace=True)

# --- Encode Categorical Variables ---
def encode_and_save(column, name):
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    pickle.dump(le, open(f'le_{name}.pkl', 'wb'))
    return data[column]

data['holiday'] = encode_and_save('holiday', 'holiday')
data['weather'] = encode_and_save('weather', 'weather')

# --- Features and Target ---
y = data['traffic_volume']
X = data.drop('traffic_volume', axis=1)

# --- Feature Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# --- Split Data ---
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Train Model ---
model = RandomForestRegressor(random_state=42)
model.fit(x_train, y_train)

# --- Evaluate Model ---
y_pred = model.predict(x_test)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# --- Save Model ---
pickle.dump(model, open('model.pkl', 'wb'))