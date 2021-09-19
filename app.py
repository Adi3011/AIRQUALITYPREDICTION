import numpy as np
from flask import Flask, request, render_template
import pickle

# Initializing App
app = Flask(__name__)

# Loading Models
random_forest_model = pickle.load(open('mlModels/random_forest.pkl', 'rb'))
knn_model = pickle.load(open('mlModels/knnr.pkl', 'rb'))
ann_model = pickle.load(open('mlModels/ann.pkl', 'rb'))

# Routes


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    model = request.form.get('model')
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features[:-1])]
    if(model == 'random_forest_prediction'):
        prediction = random_forest_model.predict(final_features)
    elif(model == 'knn_prediction'):
        prediction = knn_model.predict(final_features)
    elif(model == 'ann_prediction'):
        prediction = knn_model.predict(final_features)
    else:
        prediction = 'Something went Wrong!'
    return render_template('index.html', prediction='AQI: {:.2f}'.format(prediction[0]), predicted=True)


if __name__ == "__main__":
    app.run(debug=True)
