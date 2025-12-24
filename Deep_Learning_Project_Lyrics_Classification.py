import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import os
import random
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, GlobalAveragePooling1D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ==========================================
# 0. REPRODUCIBILITY & NLTK SETUP
# ==========================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# NLTK Stopwords indir
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

print(f"Random Seed set to: {SEED}")

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading Data...")
artists_df = pd.read_csv('artists-data.csv')
lyrics_df = pd.read_csv('lyrics-data.csv', nrows=150000) 

if 'language' in lyrics_df.columns:
    lyrics_df = lyrics_df[lyrics_df['language'] == 'en']

# ==========================================
# 2. CLEANING (Stopwords REMOVAL Added)
# ==========================================
def clean_text(text):
    # Küçük harfe çevir
    text = text.lower()
    # [Chorus], [Verse] gibi kısımları sil
    text = re.sub(r'\[.*?\]', '', text)
    # Noktalama işaretlerini ve sayıları sil
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Stopwords temizliği (Yeni eklenen kısım)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    text = " ".join(words)
    
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Cleaning Lyrics with Stopword Removal...")
lyrics_df['Lyric'] = lyrics_df['Lyric'].apply(clean_text)
# Temizlik sonrası boş kalan satırları temizle
lyrics_df = lyrics_df[lyrics_df['Lyric'].str.strip().astype(bool)]

# ==========================================
# 3. MERGE & FILTER GENRES
# ==========================================
df = lyrics_df.merge(artists_df, left_on='ALink', right_on='Link')
df['Main_Genre'] = df['Genres'].apply(lambda x: x.split(';')[0] if isinstance(x, str) else x)

top_genres = ['Pop', 'Rock', 'Hip Hop', 'Gospel/Religioso', 'Country']
df = df[df['Main_Genre'].isin(top_genres)]

print(f"Total Songs after filtering: {len(df)}")

# ==========================================
# 4. SPLIT DATA
# ==========================================
print("Splitting Data (Stratified)...")
X_raw = df['Lyric'].values
Y_raw = pd.get_dummies(df['Main_Genre']).values
genre_names = pd.get_dummies(df['Main_Genre']).columns

X_train_text, X_test_text, Y_train, Y_test = train_test_split(
    X_raw, Y_raw, test_size=0.15, random_state=SEED, stratify=Y_raw
)

# ==========================================
# 5. COMPUTE CLASS WEIGHTS
# ==========================================
y_integers = np.argmax(Y_train, axis=1)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_integers),
    y=y_integers
)
class_weight_dict = dict(enumerate(class_weights))

# ==========================================
# 6. TOKENIZATION & PADDING
# ==========================================
MAX_NB_WORDS = 30000 
MAX_LEN = 250 
EMBED_DIM = 100 

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(X_train_text)

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=MAX_LEN, padding='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_LEN, padding='post')

# ==========================================
# 7. LOAD GLOVE EMBEDDINGS
# ==========================================
print("Loading GloVe...")
embeddings_index = {}
try:
    with open('glove.6B.100d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f'Found {len(embeddings_index)} GloVe vectors.')
except FileNotFoundError:
    print("Warning: GloVe file not found. Training from scratch.")

word_index = tokenizer.word_index
num_words = min(MAX_NB_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBED_DIM))

for word, i in word_index.items():
    if i >= MAX_NB_WORDS: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# ==========================================
# 8. BUILD MODEL
# ==========================================
print("Building Model...")
model = Sequential([
    Embedding(num_words, EMBED_DIM, 
              weights=[embedding_matrix] if embeddings_index else None,
              input_length=MAX_LEN, 
              trainable=True, # Fine-tuning açık
              mask_zero=True),
    
    SpatialDropout1D(0.4),
    
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.3)),
    
    GlobalAveragePooling1D(), # Genre tespiti için tematik özetleme sağlar
    BatchNormalization(),
    
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(top_genres), activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# ==========================================
# 9. TRAINING
# ==========================================
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

print("Starting Training...")
history = model.fit(
    X_train, Y_train, 
    epochs=20, 
    batch_size=64, 
    validation_data=(X_test, Y_test), 
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight_dict
)

# ==========================================
# 10. EVALUATION & VISUALIZATION
# ==========================================
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"\nOverall Accuracy: {acc*100:.2f}%")

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(Y_test, axis=1)

print("\n--- Detailed Classification Report ---")
print(classification_report(y_true, y_pred, target_names=genre_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=genre_names, yticklabels=genre_names, cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Genre')
plt.xlabel('Predicted Genre')
plt.show()

# Training History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Doğrulama')
plt.title('Model Doğruluğu')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim')
plt.plot(history.history['val_loss'], label='Doğrulama')
plt.title('Model Kaybı (Loss)')
plt.legend()
plt.show()

# ==========================================
# 11. LIVE PREDICTION FUNCTION
# ==========================================
def predict_genre(text, top_n=3):
    cleaned = clean_text(text)
    if not cleaned: return [("Unknown", 0.0)]
    
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    
    pred = model.predict(padded, verbose=0)[0]
    
    results = [ (genre_names[i], prob * 100) for i, prob in enumerate(pred) ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]

# Test Prediction
print("\n--- Live Test ---")
test_samples = [
    "I got my truck and my beer on the dirt road",
    "Jesus loves you, don't forget to pray"
]

for sample in test_samples:
    res = predict_genre(sample)
    print(f"Lyric: '{sample}'")
    print(f"Top Prediction: {res[0][0]} ({res[0][1]:.2f}%)\n")