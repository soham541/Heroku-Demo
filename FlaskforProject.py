import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle  # To read pickle file

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('Index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]  # Take values
    final_features = [np.array(int_features)]  # Converting to array and give to prediction
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('Index.html', prediction_text="User has to pay premium of Rs. {} ".format(output))


if __name__ == "__main__":
    app.run(debug=True)
