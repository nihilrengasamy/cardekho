from flask import Flask, request, render_template
import pickle
import numpy as np

# Load your trained model (assuming it is saved as 'car_price_model.pkl')
with open('classifier.pkl', 'rb') as model_final:
    model = pickle.load(model_final)

app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('home.html')  # Main page with form input

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    try:
         fueltype = float(request.form['fueltype'])
         aspiration = float(request.form['aspiration'])
         doornumber = float(request.form['doornumber'])
         carbody = float(request.form['carbody'])
         drivewheel = float(request.form['drivewheel'])
         enginelocation = float(request.form['enginelocation'])
         carlength = float(request.form['carlength'])
         carwidth = float(request.form['carwidth'])
         carheight = float(request.form['carheight'])
         curbweight = float(request.form['curbweight'])
         enginetype = float(request.form['enginetype'])
         horsepower = float(request.form['horsepower'])
         peakrpm = float(request.form['peakrpm'])
        # Add other inputs as per your model features

        # Create a feature array in the same order your model was trained on
         features = np.array([['fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','carlength','carwidth','carheight','curbweight','enginetype','horsepower','peakrpm']])
    
        # Make prediction using the loaded model
         prediction = model.predict(features)
        
        # Format the prediction result
         result = f"${prediction[0]:,.2f}"
         return render_template('home.html', prediction_text=f'Predicted Car Price: {result}')
    except Exception as e:
         return render_template('home.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)