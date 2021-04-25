from flask import Flask, render_template, url_for, request
import numpy as np
import tensorflow as tf
import tensorflow_text as tftext
#from tensorflow_text.python import metrics
#from tensorflow_text.python.metrics.text_similarity_metric_ops import *
import emoji
import pip

filename = "sentiment_analysis_using_albert"
model = tf.saved_model.load(filename)
classNames = {0: 'neutral', 1: 'anger',2: 'fear',3: 'joy',4: 'sadness'}

def install(package):
    # Debugging
    # pip.main(["install", "--pre", "--upgrade", "--no-index",
    #         "--find-links=.", package, "--log-file", "log.txt", "-vv"])
    pip.main(["install", "--upgrade", "--no-index", "--find-links=.", package])
install("tensorflow_text")

def model_predict(input_sentence, model):
    preds = model(tf.constant(input_sentence))
    return preds


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():

    if request.method == "POST":
        message = request.form["message"]
        data = [message]
        prediction = model_predict(data, model)
        prediction = np.argmax(prediction)
        prediction = classNames[prediction]
        emotions = {'neutral' : emoji.emojize(':expressionless:'), 
        'anger' : emoji.emojize(':angry:'), 
        'fear' : emoji.emojize(':fearful:'), 
        'joy' : emoji.emojize(':smile:'), 
        'sadness' : emoji.emojize(':pensive:')}

    return render_template("predict.html", out=data[0]+"?"+prediction)


if __name__ == "__main__":
    
    app.run(debug=True)
