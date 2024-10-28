# # from flask import Flask, render_template, request
# # import pickle
# # import numpy as np

# # # Load the trained SVM model
# # with open('diabetes_svm_model.pkl', 'rb') as file:
# #     svm_model = pickle.load(file)

# # # Initialize Flask app
# # app = Flask(__name__)

# # # Route for the home page (form input)
# # @app.route('/')
# # def home():
# #     return '''
# #     <h1>Diabetes Prediction</h1>
# #     <form action="/predict" method="post">
# #         <label>Pregnancies:</label><input type="number" name="pregnancies" required><br>
# #         <label>Glucose:</label><input type="number" name="glucose" required><br>
# #         <label>Blood Pressure:</label><input type="number" name="bloodpressure" required><br>
# #         <label>Skin Thickness:</label><input type="number" name="skinthickness" required><br>
# #         <label>Insulin:</label><input type="number" name="insulin" required><br>
# #         <label>BMI:</label><input type="number" step="0.1" name="bmi" required><br>
# #         <label>Diabetes Pedigree Function:</label><input type="number" step="0.01" name="dpf" required><br>
# #         <label>Age:</label><input type="number" name="age" required><br>
# #         <button type="submit">Predict</button>
# #     </form>
# #     '''

# # # Route for prediction
# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     if request.method == 'POST':
# #         # Retrieve form data
# #         pregnancies = float(request.form['pregnancies'])
# #         glucose = float(request.form['glucose'])
# #         bloodpressure = float(request.form['bloodpressure'])
# #         skinthickness = float(request.form['skinthickness'])
# #         insulin = float(request.form['insulin'])
# #         bmi = float(request.form['bmi'])
# #         dpf = float(request.form['dpf'])
# #         age = float(request.form['age'])
        
# #         # Arrange the inputs into an array for prediction
# #         input_features = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]])
        
# #         # Make a prediction
# #         prediction = svm_model.predict(input_features)[0]
        
# #         # Output result
# #         result = "has diabetes" if prediction == 1 else "does not have diabetes"
# #         return f'The person {result}'

# # # Run the app
# # if __name__ == "__main__":
# #     app.run(debug=True)

from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained SVM model, scaler, and polynomial transformer
with open('diabetes_svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('poly_transformer.pkl', 'rb') as poly_file:
    poly_transformer = pickle.load(poly_file)

# Initialize Flask app
app = Flask(__name__)

# Route for the home page (form input)
@app.route('/')
def home():
    return '''
    <h1>Diabetes Prediction</h1>
    <form action="/predict" method="post">
        <label>Pregnancies:</label><input type="number" name="pregnancies" required><br>
        <label>Glucose:</label><input type="number" name="glucose" required><br>
        <label>Blood Pressure:</label><input type="number" name="bloodpressure" required><br>
        <label>Skin Thickness:</label><input type="number" name="skinthickness" required><br>
        <label>Insulin:</label><input type="number" name="insulin" required><br>
        <label>BMI:</label><input type="number" step="0.1" name="bmi" required><br>
        <label>Diabetes Pedigree Function:</label><input type="number" step="0.01" name="dpf" required><br>
        <label>Age:</label><input type="number" name="age" required><br>
        <button type="submit">Predict</button>
    </form>
    '''

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        bloodpressure = float(request.form['bloodpressure'])
        skinthickness = float(request.form['skinthickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = float(request.form['age'])

        # Arrange the inputs into an array for prediction
        input_features = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]])

        # Apply polynomial transformation
        input_features_poly = poly_transformer.transform(input_features)

        # Scale the input data
        input_scaled = scaler.transform(input_features_poly)

        # Make a prediction
        prediction = svm_model.predict(input_scaled)[0]

        # Output result
        result = "has diabetes" if prediction == 1 else "does not have diabetes"
        return f'The person {result}'

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

