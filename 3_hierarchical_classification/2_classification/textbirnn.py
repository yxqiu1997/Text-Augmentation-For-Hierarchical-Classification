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
            python textbirnn.py raw
        Using training set augmented by deletion/insertion/swap/synonyms:
            python textbirnn.py deletion/insertion/swap/synonyms
        Using training set augmented by contextual BERT model:
            python textbirnn.py cbert
        Using training set augmented by tf-idf algorithm:
            python textbirnn.py tfidf
        Using training set augmented by fasttext model:
            python textbirnn.py fasttext
        Using training set augmented at character level:
            python textbirnn.py cDeletion/cInsertion/cKeyboard/cOcr/cSubstitution/cSwap
        Using training set augmented by BERT/DistilBERT/RoBERTA at word generation level:
            python textbirnn.py bert/distilbert/roberta
        Using training set augmented by GPT-2/XLNet at sentence generation level:
            python textbirnn.py gpt2/xlnet
        Using training set augmented by back translation:
            python textbirnn.py sequence2sequence/transformer/t5
'''

JOB_NAME = sys.argv[1]
base_dir = os.path.dirname(__file__)
EMBEDDING_FILE = os.path.join(base_dir, 'data', 'glove.6B.50d.txt')
TRAIN_CONTENT = os.path.join(base_dir, 'data', JOB_NAME, 'train-content.txt')
TRAIN_LABEL = os.path.join(base_dir, 'data', JOB_NAME, 'train-label.txt')
TEST_CONTENT = os.path.join(base_dir, 'data', 'test', 'test-content.txt')
TEST_LABEL = os.path.join(base_dir, 'data', 'test', 'test-label.txt')
CHECKPOINTS_DIR = os.path.join(base_dir, 'checkpoints')

MAX_FEATURE = 5000
MAX_LENGTH = 400
EMBEDDING_DIMENSIONS = 50

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

# Build model
model = TextBiRNN(MAX_LENGTH, MAX_FEATURE, EMBEDDING_DIMENSIONS)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

# Fit
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max')

ck_callback = ModelCheckpoint('./checkpoints/textbirnn/weights.{epoch:02d}-{val_f1:.4f}.hdf5',
                              monitor='val_f1',
                              mode='max', verbose=2,
                              save_best_only=True,
                              save_weights_only=True)
tb_callback = TensorBoard(log_dir='./logs/textbirnn')

model.fit(train_padded_seqs, train_labels,
          batch_size=32, epochs=50,
          callbacks=[early_stopping,
                     Metrics(valid_data=(test_padded_seqs, test_labels)),
                     ck_callback, tb_callback],
          validation_data=(test_padded_seqs, test_labels),
          verbose=0)

model_name = 'textbirnn-model-' + JOB_NAME + '.best'
model_dir = os.path.join(base_dir, 'models', 'textbirnn', model_name)
model.save(model_dir)
