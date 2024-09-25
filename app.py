import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("D:\CI\SECIS6005_20301691_CLBSCSD2720\system files\steel_plate_defect_model_1_1.joblib")

# Streamlit title
st.title('Steel Plate Defect Prediction')

# Input fields for user data (27 features)
X_Minimum = st.number_input('X_Minimum', min_value=0.0)
X_Maximum = st.number_input('X_Maximum', min_value=0.0)
Y_Minimum = st.number_input('Y_Minimum', min_value=0.0)
Y_Maximum = st.number_input('Y_Maximum', min_value=0.0)
Pixels_Areas = st.number_input('Pixels_Areas', min_value=0.0)
X_Perimeter = st.number_input('X_Perimeter', min_value=0.0)
Y_Perimeter = st.number_input('Y_Perimeter', min_value=0.0)
Sum_of_Luminosity = st.number_input('Sum_of_Luminosity', min_value=0.0)
Minimum_of_Luminosity = st.number_input('Minimum_of_Luminosity', min_value=0.0)
Maximum_of_Luminosity = st.number_input('Maximum_of_Luminosity', min_value=0.0)
Length_of_Conveyer = st.number_input('Length_of_Conveyer', min_value=0.0)
TypeOfSteel_A300 = st.number_input('TypeOfSteel_A300', min_value=0.0, max_value=1.0, step=1.0)
TypeOfSteel_A400 = st.number_input('TypeOfSteel_A400', min_value=0.0, max_value=1.0, step=1.0)
Steel_Plate_Thickness = st.number_input('Steel_Plate_Thickness', min_value=0.0)
Edges_Index = st.number_input('Edges_Index', min_value=0.0)
Empty_Index = st.number_input('Empty_Index', min_value=0.0)
Square_Index = st.number_input('Square_Index', min_value=0.0)
Outside_X_Index = st.number_input('Outside_X_Index', min_value=0.0)
Edges_X_Index = st.number_input('Edges_X_Index', min_value=0.0)
Edges_Y_Index = st.number_input('Edges_Y_Index', min_value=0.0)
Outside_Global_Index = st.number_input('Outside_Global_Index', min_value=0.0)
LogOfAreas = st.number_input('LogOfAreas', min_value=0.0)
Log_X_Index = st.number_input('Log_X_Index', min_value=0.0)
Log_Y_Index = st.number_input('Log_Y_Index', min_value=0.0)
Orientation_Index = st.number_input('Orientation_Index', min_value=0.0)
Luminosity_Index = st.number_input('Luminosity_Index', min_value=0.0)
SigmoidOfAreas = st.number_input('SigmoidOfAreas', min_value=0.0)

# Button to make predictions
if st.button('Predict Defect'):
    # Gather input data into an array
    features = np.array([
        X_Minimum, X_Maximum, Y_Minimum, Y_Maximum, Pixels_Areas, X_Perimeter, Y_Perimeter,
        Sum_of_Luminosity, Minimum_of_Luminosity, Maximum_of_Luminosity, Length_of_Conveyer,
        TypeOfSteel_A300, TypeOfSteel_A400, Steel_Plate_Thickness, Edges_Index, Empty_Index,
        Square_Index, Outside_X_Index, Edges_X_Index, Edges_Y_Index, Outside_Global_Index,
        LogOfAreas, Log_X_Index, Log_Y_Index, Orientation_Index, Luminosity_Index, SigmoidOfAreas
    ]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    # Display the result
    defect_classes = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other Defects']
    result = [defect_classes[i] for i, pred in enumerate(prediction[0]) if pred == 1]
    st.success(f'The predicted defect types are: {result}')
