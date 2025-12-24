# Music Genre Classification with Deep Learning

This project implements a **Bidirectional LSTM** neural network to classify song lyrics into five distinct genres.

## Project Overview
- **Objective:** Predict music genres (Pop, Rock, Hip Hop, Gospel, Country) based on lyric content.
- **Model:** Bi-LSTM architecture with Dropout and Batch Normalization.
- **Features:** Uses pre-trained **GloVe** word embeddings and NLTK for text preprocessing.

## Key Results
The model reached a test accuracy of approximately **63-64%**. The evaluation shows high performance in Hip Hop and Rock, with some overlap observed between Pop and Dance genres.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Download `glove.6B.100d.txt` to the project root.
3. Run the script: `python Deep_Learning_Project_Lyrics_Classification.py`
