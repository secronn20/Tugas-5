import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, redirect
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from wordcloud import WordCloud
import pandas as pd

app = Flask(__name__)

# pastikan folder static ada
os.makedirs("static", exist_ok=True)

# load model
model = load_model("model_sentimen_agaklaen.h5")

# load encoder
with open("label_encoder.pickle", "rb") as f:
    encoder = pickle.load(f)

# load tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# history penyimpanan
history = []

def generate_chart():
    if len(history) == 0:
        return

    df = pd.DataFrame(history, columns=["teks","sentimen"])

    # BAR CHART
    plt.figure(figsize=(6,4))
    df["sentimen"].value_counts().plot(kind="bar")
    plt.title("Distribusi Sentimen")
    plt.xlabel("Sentimen")
    plt.ylabel("Jumlah")
    plt.savefig("static/bar.png")
    plt.close()

    # PIE CHART
    plt.figure(figsize=(6,4))
    df["sentimen"].value_counts().plot(kind="pie", autopct="%1.0f%%")
    plt.ylabel("")
    plt.title("Persentase Sentimen")
    plt.savefig("static/pie.png")
    plt.close()

    # WORDCLOUD
    text = " ".join(df["teks"].astype(str))
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    wc.to_file("static/wordcloud.png")


@app.route("/", methods=["GET","POST"])
def index():
    hasil = None

    if request.method == "POST":
        teks = request.form["teks"]

        seq = tokenizer.texts_to_sequences([teks])
        padded = pad_sequences(seq, maxlen=50)

        pred = model.predict(padded)
        hasil_index = np.argmax(pred)

        hasil = encoder.inverse_transform([hasil_index])[0]

        history.append([teks, hasil])
        generate_chart()

    return render_template("index.html", hasil=hasil, history=history)


@app.route("/hapus")
def hapus():
    history.clear()
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)