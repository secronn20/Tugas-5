import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from wordcloud import WordCloud
import pickle
import os

# pastikan folder static ada
os.makedirs("static", exist_ok=True)

# load dataset
df = pd.read_csv("dataset_lstm_agaklaen.csv")

# hapus data kosong
df = df.dropna(subset=['stemming','sentimen'])

texts = df['stemming'].astype(str)
labels = df['sentimen']

# encoding label
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# tokenizing
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# padding
X = pad_sequences(sequences, maxlen=50)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# model LSTM
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=50))
model.add(LSTM(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# training
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ======================
# GRAFIK ACCURACY
# ======================
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.legend(['Train','Validation'])
plt.show()

# ======================
# GRAFIK LOSS
# ======================
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.legend(['Train','Validation'])
plt.show()

# ======================
# CONFUSION MATRIX
# ======================
y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ======================
# DISTRIBUSI SENTIMEN CSV
# ======================
plt.figure()
df['sentimen'].value_counts().plot(kind='bar')
plt.title("Distribusi Sentimen Dataset")
plt.show()

# ======================
# WORDCLOUD POSITIF
# ======================
text_pos = " ".join(df[df['sentimen']=="positif"]['stemming'].astype(str))
wc_pos = WordCloud(width=800,height=400,background_color="white").generate(text_pos)

plt.figure()
plt.imshow(wc_pos)
plt.axis("off")
plt.title("WordCloud Positif")
plt.show()

# ======================
# WORDCLOUD NETRAL
# ======================
text_net = " ".join(df[df['sentimen']=="netral"]['stemming'].astype(str))
wc_net = WordCloud(width=800,height=400,background_color="white").generate(text_net)

plt.figure()
plt.imshow(wc_net)
plt.axis("off")
plt.title("WordCloud Netral")
plt.show()

# ======================
# WORDCLOUD NEGATIF
# ======================
text_neg = " ".join(df[df['sentimen']=="negatif"]['stemming'].astype(str))
wc_neg = WordCloud(width=800,height=400,background_color="white").generate(text_neg)

plt.figure()
plt.imshow(wc_neg)
plt.axis("off")
plt.title("WordCloud Negatif")
plt.show()

# simpan model
model.save("model_sentimen_agaklaen.h5")

# simpan tokenizer
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# simpan encoder
with open("label_encoder.pickle", "wb") as f:
    pickle.dump(encoder, f)

print("Training selesai")