import sys

print('python executable:', sys.executable)

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    print('tensorflow version:', tf.__version__)
    print('Tokenizer import: OK')
except Exception as e:
    print('IMPORT ERROR:', type(e).__name__, str(e))
