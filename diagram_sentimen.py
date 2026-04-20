import pandas as pd
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv("dataset_lstm_agaklaen.csv")

# hapus data kosong
df = df.dropna(subset=['sentimen'])

# hitung jumlah sentimen
sentimen_count = df['sentimen'].value_counts()

print(sentimen_count)

# plot diagram batang
plt.figure()
sentimen_count.plot(kind='bar')

plt.title("Distribusi Sentimen Film Agak Laen")
plt.xlabel("Sentimen")
plt.ylabel("Jumlah")
plt.show()

plt.figure()
sentimen_count.plot(kind='pie', autopct='%1.1f%%')
plt.title("Persentase Sentimen")
plt.ylabel("")
plt.show()