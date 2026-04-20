import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# load model
model = load_model("model_sentimen_agaklaen.h5")

# load dataset untuk ambil label encoder
df = pd.read_csv("dataset_lstm_agaklaen.csv")
df = df.dropna(subset=['stemming','sentimen'])

texts = df['stemming'].astype(str)
labels = df['sentimen']

# encoding label
encoder = LabelEncoder()
encoder.fit(labels)

# tokenizer ulang
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)

# input komentar baru
komentar = input("Masukkan komentar: ")

# preprocessing sederhana
komentar = komentar.lower()

sequence = tokenizer.texts_to_sequences([komentar])
padded = pad_sequences(sequence, maxlen=50)

# prediksi
prediksi = model.predict(padded)
hasil = encoder.inverse_transform([np.argmax(prediksi)])

print("Sentimen:", hasil[0])