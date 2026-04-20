import pandas as pd
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# download stopwords
nltk.download('stopwords')

# load dataset
df = pd.read_csv("komentar_youtube_agaklaen_1.csv")

# case folding & cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['clean'] = df['komentar'].apply(clean_text)

# stopword removal
stop_words = set(stopwords.words('indonesian'))

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])

df['stopword'] = df['clean'].apply(remove_stopwords)

# stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()

df['stemming'] = df['stopword'].apply(stemmer.stem)

# simpan hasil
df.to_csv("hasil_preprocessing.csv", index=False)

print("Preprocessing selesai")