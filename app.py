from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model # type: ignore
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('tfhddFinal.h5')

@app.route('/')
def home():
    return render_template('index.html')  # Renders the HTML form
# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the POST request
    data = request.get_json()  # Expecting JSON format
    input_data = np.array(data['input'])  # Convert the input data into a numpy array (if needed)

    # Reshape if required (e.g., if the model expects a 2D input)
    #input_data = np.array(44, 0, 2, 118, 242, 0, 1, 149, 0, 0.3, 1, 1, 2);
    #input_data = np.array([51, 1, 2, 94, 227, 0, 1, 154, 1, 0, 2, 1, 3])
    input_data = input_data.reshape(1, -1)  # Adjust dimensions as necessary
    print(input_data)
   
    # Make prediction using the model
    prediction = model.predict(input_data)

    # Return the prediction result as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)