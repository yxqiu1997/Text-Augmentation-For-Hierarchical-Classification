from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from get_models import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from metrics import *
import tensorflow as tf
import numpy as np
import os
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

'''
    Command line usage:
        Using original training set:
            python textcnn.py raw
        Using training set augmented by deletion/insertion/swap/synonyms:
            python textcnn.py deletion/insertion/swap/synonyms
        Using training set augmented by contextual BERT model:
            python textcnn.py cbert
        Using training set augmented by tf-idf algorithm:
            python textcnn.py tfidf
        Using training set augmented by fasttext model:
            python textcnn.py fasttext
        Using training set augmented at character level:
            python textcnn.py cDeletion/cInsertion/cKeyboard/cOcr/cSubstitution/cSwap
        Using training set augmented by BERT/DistilBERT/RoBERTA at word generation level:
            python textcnn.py bert/distilbert/roberta
        Using training set augmented by GPT-2/XLNet at sentence generation level:
            python textcnn.py gpt2/xlnet
        Using training set augmented by back translation:
            python textcnn.py sequence2sequence/transformer/t5
'''

JOB_NAME = sys.argv[1]
base_dir = os.path.dirname(__file__)
EMBEDDING_FILE = os.path.join(base_dir, 'data', 'glove.6B.50d.txt')
EMBEDDING_SIZE = 50
TRAIN_CONTENT = os.path.join(base_dir, 'data', JOB_NAME, 'train-content.txt')
TRAIN_LABEL = os.path.join(base_dir, 'data', JOB_NAME, 'train-label.txt')
TEST_CONTENT = os.path.join(base_dir, 'data', 'test', 'test-content.txt')
TEST_LABEL = os.path.join(base_dir, 'data', 'test', 'test-label.txt')
CHECKPOINTS_DIR = os.path.join(base_dir, 'checkpoints')

MAX_FEATURE = 5000
MAX_LENGTH = 400
FILTERS_NUM = 32

train_contents = []
with open(TRAIN_CONTENT, 'r', encoding='utf-8') as r:
    lines = r.readlines()
    for line in lines:
        train_contents.append(line.strip())

train_labels = []
with open(TRAIN_LABEL, 'r', encoding='utf-8') as r:
    lines = r.readlines()
    for line in lines:
        label = int(line[9: 10])
        array = np.zeros(4, dtype=np.int)
        array[label - 1] = 1
        train_labels.append(array)
train_labels = np.array(train_labels)

test_contents = []
with open(TEST_CONTENT, 'r', encoding='utf-8') as r:
    lines = r.readlines()
    for line in lines:
        test_contents.append(line.strip())

test_labels = []
with open(TEST_LABEL, 'r', encoding='utf-8') as r:
    lines = r.readlines()
    for line in lines:
        label = int(line[9: 10])
        array = np.zeros(4, dtype=np.int)
        array[label - 1] = 1
        test_labels.append(array)
test_labels = np.array(test_labels)

tokenizer = Tokenizer(num_words=MAX_FEATURE,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                      lower=True)
tokenizer.fit_on_texts(train_contents)
# vocab = tokenizer.word_index

train_word_ids = tokenizer.texts_to_sequences(train_contents)
test_word_ids = tokenizer.texts_to_sequences(test_contents)

train_padded_seqs = pad_sequences(train_word_ids, maxlen=MAX_LENGTH)
test_padded_seqs = pad_sequences(test_word_ids, maxlen=MAX_LENGTH)

# Word embedding
embedding_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE, encoding='utf-8'))
all_embeddings = np.stack(embedding_index.values())
embedding_mean, embedding_std = all_embeddings.mean(), all_embeddings.std()
word_index = tokenizer.word_index
nb_words = min(MAX_FEATURE, len(word_index))
embedding_matrix = np.random.normal(embedding_mean, embedding_std,
                                    (nb_words, EMBEDDING_SIZE))
for word, i in word_index.items():
    if i >= MAX_FEATURE:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i - 1] = embedding_vector

model = get_textcnn_model(MAX_LENGTH, MAX_FEATURE, EMBEDDING_SIZE, embedding_matrix)

# Fit
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max')

ck_callback = ModelCheckpoint('./checkpoints/textcnn/weights.{epoch:02d}-{val_f1:.4f}.hdf5',
                              monitor='val_f1',
                              mode='max', verbose=2,
                              save_best_only=True,
                              save_weights_only=True)
tb_callback = TensorBoard(log_dir='./logs/textcnn')

model.fit(train_padded_seqs, train_labels,
          batch_size=64, epochs=50,
          callbacks=[early_stopping,
                     Metrics(valid_data=(test_padded_seqs, test_labels)),
                     ck_callback, tb_callback],
          validation_data=(test_padded_seqs, test_labels),
          verbose=0)

model_name = 'textcnn-model-' + JOB_NAME + '.best'
model_dir = os.path.join(base_dir, 'models', 'textcnn', model_name)
model.save(model_dir)
