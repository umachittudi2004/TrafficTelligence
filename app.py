"""from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/inspect')
def inspect():
    return render_template('inspect.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [int(request.form.get(field)) for field in [
            'Holiday', 'Temp', 'Rain', 'snow', 'weather', 'year', 'month', 'day', 'hours', 'minutes', 'seconds'
        ]]
        prediction = model.predict([np.array(features)])
        result = f'Predicted Traffic Volume: {prediction[0]:.2f}'
    except Exception as e:
        result = f'Error: {str(e)}'
    return render_template('inspect.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True) """
"""
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and encoder
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))  # Added this line

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/inspect')
def inspect():
    return render_template('inspect.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect raw input from form
        raw_features = [int(request.form.get(field)) for field in [
            'Holiday', 'Temp', 'Rain', 'snow', 'weather', 'year', 'month', 'day', 'hours', 'minutes', 'seconds'
        ]]
        
        # Transform input using encoder
        transformed_input = encoder.transform([raw_features])
        
        # Predict using model
        prediction = model.predict(transformed_input)
        result = f'Predicted Traffic Volume: {prediction[0]:.2f}'
    except Exception as e:
        result = f'Error: {str(e)}'
    return render_template('inspect.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
"""
"""
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('encoder.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/inspect')
def inspect():
    return render_template('inspect.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_data = [
            request.form.get('Holiday'),
            float(request.form.get('Temp')),
            float(request.form.get('Rain')),
            float(request.form.get('snow')),
            request.form.get('weather'),
            int(request.form.get('year')),
            int(request.form.get('month')),
            int(request.form.get('day')),
            int(request.form.get('hours')),
            int(request.form.get('minutes')),
            int(request.form.get('seconds'))
        ]

        # Convert to numpy array
        input_data = np.array([raw_data])

        # If holiday/weather are already integers, skip encoding
        input_data = input_data.astype(float)

        # Apply scaler
        scaled_input = scaler.transform(input_data)

        prediction = model.predict(scaled_input)
        result = f'Predicted Traffic Volume: {prediction[0]:.2f}'
    except Exception as e:
        result = f'Error: {str(e)}'
    return render_template('inspect.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
    
"""

# app.py

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and preprocessing tools
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le_holiday = pickle.load(open('le_holiday.pkl', 'rb'))
le_weather = pickle.load(open('le_weather.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        print(f"Received contact message from {name} ({email}): {message}")
        return render_template('contact.html', message_sent=True,user_name=name)
    return render_template('contact.html')

@app.route('/inspect')
def inspect():
    return render_template('inspect.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve raw inputs
        holiday_raw = request.form.get('Holiday')
        weather_raw = request.form.get('weather')

        # Ensure fallback values to avoid LabelEncoder errors
        if holiday_raw not in le_holiday.classes_:
            holiday_raw = 'None'  # Default holiday class
        if weather_raw not in le_weather.classes_:
            weather_raw = 'Clouds'  # Default weather class

        # Encode
        holiday = le_holiday.transform([holiday_raw])[0]
        weather = le_weather.transform([weather_raw])[0]

        # Convert numeric inputs
        temp = float(request.form['Temp'])
        rain = float(request.form['Rain'])
        snow = float(request.form['snow'])
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hours = int(request.form['hours'])
        minutes = int(request.form['minutes'])
        seconds = int(request.form['seconds'])

        # Final feature vector
        features = [holiday, temp, rain, snow, weather, year, month, day, hours, minutes, seconds]
        scaled_features = scaler.transform([features])

        # Make prediction
        prediction = model.predict(scaled_features)
        result = f'Predicted Traffic Volume: {prediction[0]:,.2f}'

    except Exception as e:
        result = f'Error: {str(e)}'

    # Render separate result page
    return render_template('result.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)