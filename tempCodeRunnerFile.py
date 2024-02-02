import gradio as gr
import joblib
import json

# Load the model
model_path = 'D:\Machine Learning\MLProjects\Project\Real estate project\\banglore_home_prices_model.joblib'
columns_path = 'D:\Machine Learning\MLProjects\Project\Real estate project\columns.json'

with open(model_path, 'rb') as f:
    model = joblib.load(f)

# Load columns information
with open(columns_path, 'r') as f:
    locations = json.load(f)['data_columns']

# Define a function for prediction
def predict_price(location, sqft, bath, bhk):
    # Perform prediction using the loaded model
    print(f"Inputs: {location}, {sqft}, {bath}, {bhk}")
    result = model.predict([[sqft, bath, bhk]])[0].round()
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
