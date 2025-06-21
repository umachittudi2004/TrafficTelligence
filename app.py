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

from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load model and preprocessing tools
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('encoder.pkl', 'rb'))
le_holiday = pickle.load(open('le_holiday.pkl', 'rb'))
le_weather = pickle.load(open('le_weather.pkl', 'rb'))

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
        # Get form inputs
        holiday = request.form.get('Holiday')            # e.g. "None"
        temp = float(request.form.get('Temp'))
        rain = float(request.form.get('Rain'))
        snow = float(request.form.get('snow'))
        weather = request.form.get('weather')            # e.g. "Clouds"
        year = int(request.form.get('year'))
        month = int(request.form.get('month'))
        day = int(request.form.get('day'))
        hour = int(request.form.get('hours'))
        minutes = int(request.form.get('minutes'))
        seconds = int(request.form.get('seconds'))

        # Encode holiday and weather
        holiday_encoded = le_holiday.transform([holiday])[0]
        weather_encoded = le_weather.transform([weather])[0]

        # Prepare final input
        input_data = np.array([[holiday_encoded, temp, rain, snow, weather_encoded,
                                year, month, day, hour, minutes, seconds]])

        # Scale input
        scaled_input = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_input)
        result = f"Predicted Traffic Volume: {prediction[0]:.2f}"

    except Exception as e:
        result = f"Error: {str(e)}"

    # Redirect to result page with the prediction
    return redirect(url_for('result', prediction_text=result))

@app.route('/result')
def result():
    prediction_text = request.args.get('prediction_text', 'No result available.')
    return render_template('result.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
