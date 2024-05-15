import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump

# Load dataset
data = pd.read_csv("./dataset.csv")

# Split data into features and labels
X = data.drop("Accident_severity", axis=1)  # Drop the target variable
y = data["Accident_severity"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical and numerical features
categorical_features = ['Age_band_of_driver', 'Sex_of_driver', 'Educational_level', 
                        'Vehicle_driver_relation', 'Driving_experience', 'Lanes_or_Medians', 
                        'Types_of_Junction', 'Road_surface_type', 'Light_conditions', 
                        'Weather_conditions', 'Type_of_collision', 'Vehicle_movement', 
                        'Pedestrian_movement', 'Cause_of_accident']
numeric_features = []

# Preprocessing for numerical features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

# Preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Initialize Logistic Regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save the trained model
dump(model, 'logistic_regression_model.joblib')
