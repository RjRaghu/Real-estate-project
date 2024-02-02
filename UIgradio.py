import joblib
import json
import numpy as np
import gradio as gr

# Load the model
model_path = 'D:\Machine Learning\MLProjects\Project\Real estate project\\banglore_home_prices_model.joblib'
columns_path = 'D:\Machine Learning\MLProjects\Project\Real estate project\columns.json'

with open(model_path, 'rb') as f:
    model = joblib.load(f)

# Load columns information
with open(columns_path, 'r') as f:
    locations = json.load(f)['data_columns']

# Get the number of features
num_features = len(locations) + 3  # 3 additional features for sqft, bath, bhk

# Define a function for prediction
def predict_price(location, sqft, bath, bhk):
    # Perform prediction using the loaded model
    print(f"Inputs: {location}, {sqft}, {bath}, {bhk}")

    # Create an array with zeros for all features
    x = np.zeros(num_features)

    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    # Find the index corresponding to the provided location
    loc_index = np.where(np.array(locations) == location)[0][0]

    if loc_index >= 0:
        # Set the value for the location feature
        x[loc_index + 3] = 1  # 3 is the offset for sqft, bath, bhk features

    result = model.predict([x])[0].round()
    return f"Predicted Price: {result} Lakhs"

# Create a Gradio interface
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Dropdown(locations, label="Location"),
        gr.Number(minimum=1000, label="Total Square Feet Area"),
        gr.Number(minimum=1, maximum=10, label="Number of Bathrooms"),
        gr.Number(minimum=1, maximum=10, label="Number of Bedrooms")
    ],
    outputs=gr.Textbox()
)

# Launch the interface
iface.launch()
