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
    return prediction[0]

# Routes

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/show', methods=['POST'])
def show():
    # fig = go.Figure(go.Indicator(
    #     mode = "gauge+number+delta",
    #     value = 420,
    #     domain = {'x': [0, 1], 'y': [0, 1]},
    #     title = {'text': "Speed", 'font': {'size': 24}},
    #     delta = {'reference': 400, 'increasing': {'color': "RebeccaPurple"}},
    #     gauge = {
    #     'axis': {'range': [None, 500], 'tickwidth': 1, 'tickcolor': "darkblue"},
    #     'bar': {'color': "darkblue"},
    #     'bgcolor': "white",
    #     'borderwidth': 2,
    #     'bordercolor': "gray",
    #     'steps': [
    #         {'range': [0, 250], 'color': 'cyan'},
    #         {'range': [250, 400], 'color': 'royalblue'}],
    #     'threshold': {
    #         'line': {'color': "red", 'width': 4},
    #         'thickness': 0.75,
    #         'value': 490}}))

    # fig.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})
    # # fig.show()
    # df = pd.DataFrame({
    #   "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    #   "Amount": [4, 1, 2, 2, 4, 5],
    #   "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    # })
   
    # fig = px.bar(df, x="Fruit", y="Amount", color="City",    barmode="group")
    # graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # return render_template('index.html', graphJSON=graphJSON)
    bar = create_plot()
    return render_template('index.html', plot=bar)
    # prediction = predict()
    # return render_template('index.html', prediction='AQI: {:.2f}'.format(prediction), predicted=True)

def create_plot():


    N = 40
    x = np.linspace(0, 1, N)
    y = np.random.randn(N)
    df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe


    data = [
        go.Bar(
            x=df['x'], # assign x as the dataframe column 'x'
            y=df['y']
        )
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)


if __name__ == "__main__":
    app.run(debug=True)
