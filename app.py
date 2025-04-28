from flask import Flask, render_template, request, jsonify
import pickle, pandas as pd, json, os

app = Flask(__name__)
# Load model and metrics
model = pickle.load(open('model.pkl', 'rb'))
with open('metrics.json') as f:
    metrics = json.load(f)
# Load data for charts
df = pd.read_csv('data.csv')

@app.route('/')
def home():
    return render_template('index.html', metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    t = float(request.form['tariff'])
    v = float(request.form['volume'])
    p = model.predict([[t, v]])[0]
    return render_template('index.html', prediction_text=f"Predicted Price: ${p:.2f}", metrics=metrics)

@app.route('/data')
def data():
    return jsonify(df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)