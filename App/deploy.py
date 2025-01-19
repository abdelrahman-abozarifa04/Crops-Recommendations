from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load all models into a dictionary
models = [
    'GaussianNB',
    'DecisionTreeClassifier',
    'RandomForestClassifier',
    'ExtraTreesClassifier'
]

loaded_models = {}

# Load each model from the corresponding file
for model_name in models:
    filename = f'{model_name}_savedmodel.sav'
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
    # Store the loaded model in the dictionary
    loaded_models[model_name] = loaded_model

# Load the label encoder (assuming it was saved during training)
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html', result=None)


@app.route('/predict', methods=['POST'])
def predict():
    # Get inputs from the user
    Nitrogen = float(request.form['Nitrogen'])
    Phosphorus = float(request.form['Phosphorus'])
    Potassium = float(request.form['Potassium'])
    Temperature = float(request.form['Temperature'])
    Humidity = float(request.form['Humidity'])
    pH = float(request.form['pH'])
    Annual_rainfall = float(request.form['Annual_rainfall'])

    # Prepare input data for prediction
    input_data = np.array([[Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Annual_rainfall]])

    # Dictionary to store predictions from each model
    predictions = {}

    # Iterate over each model to make predictions
    for model_name, model in loaded_models.items():
        prediction = model.predict(input_data)
        predicted_crop_index = prediction[0]
        predicted_crop = label_encoder.inverse_transform([predicted_crop_index])[0]
        predictions[model_name] = predicted_crop

    # Pass the predictions and input values to the template
    return render_template('index.html',
                           result=predictions,
                           Nitrogen=Nitrogen,
                           Phosphorus=Phosphorus,
                           Potassium=Potassium,
                           Temperature=Temperature,
                           Humidity=Humidity,
                           pH=pH,
                           Annual_rainfall=Annual_rainfall)


if __name__ == '__main__':
    app.run(debug=True)
