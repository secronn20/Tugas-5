import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# load dataset
df = pd.read_csv("dataset_lstm_agaklaen.csv")

# hapus data kosong
df = df.dropna(subset=['stemming'])

# gabungkan semua teks
text = " ".join(df['stemming'].astype(str))

# buat wordcloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(text)

# tampilkan
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud Komentar Film Agak Laen")
plt.show()