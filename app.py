from flask import Flask, render_template, request
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from joblib import load


app = Flask(__name__)

# Load model
model = load('logistic_regression_model.joblib')  # Load the trained model here

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from form
        Age_band_of_driver = request.form['Age_band_of_driver']
        Sex_of_driver = request.form['Sex_of_driver']
        Educational_level = request.form['Educational_level']
        Vehicle_driver_relation = request.form['Vehicle_driver_relation']
        Driving_experience = request.form['Driving_experience']
        Lanes_or_Medians = request.form['Lanes_or_Medians']
        Types_of_Junction = request.form['Types_of_Junction']
        Road_surface_type = request.form['Road_surface_type']
        Light_conditions = request.form['Light_conditions']
        Weather_conditions = request.form['Weather_conditions']
        Type_of_collision = request.form['Type_of_collision']
        Vehicle_movement = request.form['Vehicle_movement']
        Pedestrian_movement = request.form['Pedestrian_movement']
        Cause_of_accident = request.form['Cause_of_accident']
        Accident_severity = request.form['Accident_severity']
        
        # Preprocess input data
        input_data = pd.DataFrame({
            'Age_band_of_driver': [Age_band_of_driver],
            'Sex_of_driver': [Sex_of_driver],
            'Educational_level': [Educational_level],
            'Vehicle_driver_relation': [Vehicle_driver_relation],
            'Driving_experience': [Driving_experience],
            'Lanes_or_Medians': [Lanes_or_Medians],
            'Types_of_Junction': [Types_of_Junction],
            'Road_surface_type': [Road_surface_type],
            'Light_conditions': [Light_conditions],
            'Weather_conditions': [Weather_conditions],
            'Type_of_collision': [Type_of_collision],
            'Vehicle_movement': [Vehicle_movement],
            'Pedestrian_movement': [Pedestrian_movement],
            'Cause_of_accident': [Cause_of_accident],
            'Accident_severity': [Accident_severity]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]

        if prediction==2:
            return "<h1>Saviour Injury</h1>"
        elif prediction==1:
            return "<h1>Serious Injury</h1>"
        
if __name__ == '__main__':
    app.run(debug=True)
