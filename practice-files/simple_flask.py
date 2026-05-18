from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def hello():
    return "<p>Hello from practice-files!</p>"

@app.route("/predict")
def predict():
    return jsonify({
        "model": "fake-ai-vi",
        "result": "cat",
        "confidence": 0.99
    })
app.run(host="0.0.0.0", port=5000)