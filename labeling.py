import pandas as pd

# load data
df = pd.read_csv("hasil_preprocessing.csv")

# kamus sederhana
positif = ['lucu', 'bagus', 'keren', 'mantap', 'seru', 'hebat', 'suka']
negatif = ['jelek', 'buruk', 'bosen', 'garing', 'ga lucu', 'kurang']
    
def label_sentimen(text):
    text = str(text)
    
    for word in positif:
        if word in text:
            return "positif"
    
    for word in negatif:
        if word in text:
            return "negatif"
    
    return "netral"

df['sentimen'] = df['stemming'].apply(label_sentimen)

df.to_csv("dataset_lstm_agaklaen.csv", index=False)

print("Labeling selesai")