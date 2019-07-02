import numpy as np
import os,re
import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from tools.helper import MetricsAtTopK, clean_str
import scipy.sparse
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# things
DATA_DIR = './data/amazon_des.pkl'
OUT_DIR = './data/amazon_xmlcnn'
EMBEDDINGS_DIR = './glove.840B.300d.txt'
MAX_SEQUENCE_LENGTH = 500
MAX_NUM_WORDS = 50000
EMBEDDING_DIM = 300

# load sentences
print('READING DATA FROM : {}'.format(DATA_DIR))
df = pd.read_pickle(DATA_DIR)
# clean text
print('CLEAN TEXT')
df['text']= df['text'].apply(clean_str)
# train val split
train_df = df[df['train/test']=='train']
test_df = df[df['train/test']=='test']
# build tokenizer
print('BUILD TOKENIZER')
tokenizer = Tokenizer(
    num_words = MAX_NUM_WORDS,
    oov_token = '<UNK>',
    lower = True)
tokenizer.fit_on_texts(train_df['text'].values) # will take very long
word_index = tokenizer.word_index
print('Found {} unique tokens.'.format(len(word_index)))
# get sequence
print('TRANSFORM TEXT TO SEQUENCE')
x_train = tokenizer.texts_to_sequences(train_df['text'].values)
x_train = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
x_test = tokenizer.texts_to_sequences(test_df['text'].values)
x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
# get labels
print('BUILD MULTILABELBINARIZER')
mlb = MultiLabelBinarizer(sparse_output=True)
mlb.fit(df['categories'].values)
y_train = mlb.transform(train_df['categories'].values)
y_test = mlb.transform(test_df['categories'].values)
print('Found {} classes'.format(len(mlb.classes_)))
# get embedding matrix
print('BUILD EMBEDDING MATRIX')
vocab = set(word_index)
embeddings_index = {}
with open(EMBEDDINGS_DIR) as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        if word in vocab:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))
# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.random.uniform(-0.25, 0.25,(num_words, EMBEDDING_DIM))
empty = 0
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        empty+=1
print('{:.2f}% vocab has word embeddings'.format((num_words-empty)/num_words*100))
labels_dim = len(mlb.classes_)
num_words = min(MAX_NUM_WORDS, len(tokenizer.word_index)) + 1

# save things
print('SAVING THINGS')
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
with open('{}/tokenizer.pkl'.format(OUT_DIR), 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('{}/mlb.pkl'.format(OUT_DIR),'wb') as f:
    pickle.dump(mlb,f, protocol=pickle.HIGHEST_PROTOCOL)
np.save('{}/embedding_matrix.npy'.format(OUT_DIR),embedding_matrix)
np.save('{}/x_train.npy'.format(OUT_DIR),x_train)
np.save('{}/x_test.npy'.format(OUT_DIR),x_test)
scipy.sparse.save_npz('{}/y_train.npz'.format(OUT_DIR),y_train)
scipy.sparse.save_npz('{}/y_test.npz'.format(OUT_DIR),y_test)
