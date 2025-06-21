# importing the necessary libraries
"""
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
import xgboost
data=pd.read_csv("traffic volume.csv")
data.head()
data.describe()
data.info()
data.isnull().sum()
data['temp'] = data['temp'].fillna(data['temp'].mean())
data['rain'] = data['rain'].fillna(data['rain'].mean())
data['snow'] = data['snow'].fillna(data['snow'].mean())

from collections import Counter

print(Counter(data['weather']))

data['weather'] = data['weather'].fillna('Clouds')
columns_for_correlation = ['temp', 'rain', 'snow', 'traffic_volume'] # Replace with your actual numerical column names
correlation_matrix = data[columns_for_correlation].corr()
sns.pairplot(data)
data.boxplot()
# splitting the date column into year,month,day
data[["day", "month", "year"]] = data["date"].str.split("-", expand = True)

# splitting the date column into year,month,day
data[["hours", "minutes", "seconds"]] = data["Time"].str.split(":", expand = True)

data.drop(columns=['date', 'Time'],axis=1,inplace=True)

data.head()

from sklearn.preprocessing import LabelEncoder

# Assuming 'some_categorical_column' is a column you want to label encode
le = LabelEncoder()
data['holiday'] = le.fit_transform(data['holiday'])
data['weather'] = le.fit_transform(data['weather'])
y = data['traffic_volume']
# Separate the target variable
data_features = data.drop('traffic_volume', axis=1)


# One-hot encode both 'holiday' and 'weather' columns


# Now scale the encoded features
names = data_features.columns
from sklearn.preprocessing import scale
encoder= scale(data_features)
x = pd.DataFrame(encoder, columns=names)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

lin_reg=linear_model.LinearRegression()
lin_reg.fit(x_train,y_train)
Dtree=tree.DecisionTreeRegressor()
Dtree.fit(x_train,y_train)
Rforest=ensemble.RandomForestRegressor()
Rforest.fit(x_train,y_train)
svr=svm.SVR()
svr.fit(x_train,y_train)
xgb=xgboost.XGBRegressor()
xgb.fit(x_train,y_train)

p1=lin_reg.predict(x_train)
p2=Dtree.predict(x_train)
p3=Rforest.predict(x_train)
p4=svr.predict(x_train)
p5=xgb.predict(x_train)

from sklearn import metrics
print(metrics.r2_score(p1,y_train))
print(metrics.r2_score(p2,y_train))
print(metrics.r2_score(p3,y_train))
print(metrics.r2_score(p4,y_train))
print(metrics.r2_score(p5,y_train))

a1=lin_reg.predict(x_test)
a2=Dtree.predict(x_test)
a3=Rforest.predict(x_test)
a4=svr.predict(x_test)
a5=xgb.predict(x_test)

print(metrics.r2_score(a1,y_test))
print(metrics.r2_score(a2,y_test))
print(metrics.r2_score(a3,y_test))
print(metrics.r2_score(a4,y_test))
print(metrics.r2_score(a5,y_test))

MSE=metrics.mean_squared_error(a3,y_test)
print(np.sqrt((MSE)))

import pickle
pickle.dump(Rforest,open('model.pkl','wb'))
pickle.dump(encoder,open('encoder.pkl','wb'))
"""

# ML_model.py

import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import xgboost  # used during development

# Load the dataset
data = pd.read_csv("traffic volume.csv")

# Fill missing values
data['temp'] = data['temp'].fillna(data['temp'].mean())
data['rain'] = data['rain'].fillna(data['rain'].mean())
data['snow'] = data['snow'].fillna(data['snow'].mean())
data['weather'] = data['weather'].fillna('Clouds')

# Extract date and time parts
data[["day", "month", "year"]] = data["date"].str.split("-", expand=True)
data[["hours", "minutes", "seconds"]] = data["Time"].str.split(":", expand=True)
data.drop(columns=['date', 'Time'], axis=1, inplace=True)

# Encode categorical features
le_holiday = LabelEncoder()
data['holiday'] = le_holiday.fit_transform(data['holiday'])

le_weather = LabelEncoder()
data['weather'] = le_weather.fit_transform(data['weather'])

# Save the encoders
pickle.dump(le_holiday, open('le_holiday.pkl', 'wb'))
pickle.dump(le_weather, open('le_weather.pkl', 'wb'))

# Prepare features and target
y = data['traffic_volume']
X = data.drop('traffic_volume', axis=1)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
pickle.dump(scaler, open('encoder.pkl', 'wb'))

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = ensemble.RandomForestRegressor()
model.fit(x_train, y_train)

# Save the model
pickle.dump(model, open('model.pkl', 'wb'))
