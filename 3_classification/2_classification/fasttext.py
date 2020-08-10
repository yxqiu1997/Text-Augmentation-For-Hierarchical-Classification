from get_models import *
from metrics import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
import os
import sys
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

'''
    Command line usage:
        Using original training set:
            python fasttext.py raw
        Using training set augmented by deletion/insertion/swap/synonyms:
            python fasttext.py deletion/insertion/swap/synonyms
        Using training set augmented by contextual BERT model:
            python fasttext.py cbert
        Using training set augmented by tf-idf algorithm:
            python fasttext.py tfidf
        Using training set augmented by fasttext model:
            python fasttext.py fasttext
        Using training set augmented at character level:
            python fasttext.py cDeletion/cInsertion/cKeyboard/cOcr/cSubstitution/cSwap
        Using training set augmented by BERT/DistilBERT/RoBERTA at word generation level:
            python fasttext.py bert/distilbert/roberta
        Using training set augmented by GPT-2/XLNet at sentence generation level:
            python fasttext.py gpt2/xlnet
        Using training set augmented by back translation:
            python fasttext.py sequence2sequence/transformer/t5
'''

JOB_NAME = sys.argv[1]
base_dir = os.path.dirname(__file__)
EMBEDDING_FILE = os.path.join(base_dir, 'data', 'glove.6B.50d.txt')
TRAIN_CONTENT = os.path.join(base_dir, 'data', JOB_NAME, 'train-content.txt')
TRAIN_LABEL = os.path.join(base_dir, 'data', JOB_NAME, 'train-label.txt')
TEST_CONTENT = os.path.join(base_dir, 'data', 'test', 'test-content.txt')
TEST_LABEL = os.path.join(base_dir, 'data', 'test', 'test-label.txt')
CHECKPOINTS_DIR = os.path.join(base_dir, 'checkpoints')


def create_ngram_set(input_list, ngram_value=2):
    """
        Extract a set of n-grams from a list of integers.
        # >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
        {(4, 9), (4, 1), (1, 4), (9, 4)}
        # >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
        [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
        Augment the input list of list (sequences) by appending n-grams values.
        Example: adding bi-gram
        # >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        # >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
        # >>> add_ngram(sequences, token_indice, ngram_range=2)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
        Example: adding tri-gram
        # >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        # >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
        # >>> add_ngram(sequences, token_indice, ngram_range=3)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

NGRAM_RANGE = 2
MAX_LENGTH = 400
EMBEDDING_DIMENSIONS = 50
max_features = 5000

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

# Create set of unique n-gram from the training set
ngram_set = set()
for input_list in train_contents:
    for i in range(2, NGRAM_RANGE + 1):
        set_of_ngram = create_ngram_set(input_list, ngram_value=i)
        ngram_set.update(set_of_ngram)

# Dictionary mapping n-gram token to a unique integer
# Integer values are greater than MAX_FEATURE in order
# to avoid collision with existing features
start_index = max_features + 1
token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
indice_token = {token_indice[k]: k for k in token_indice}

# MAX_FEATURE is the highest integer that could be found in the dataset
max_features = np.max(list(indice_token.keys())) + 1

# Augmenting train_contents adn test_contents with n-gram features
# bi-gram in this case
tokenizer = Tokenizer(num_words=max_features,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                      lower=True)
tokenizer.fit_on_texts(train_contents)

train_word_ids = tokenizer.texts_to_sequences(train_contents)
test_word_ids = tokenizer.texts_to_sequences(test_contents)

train_contents = add_ngram(train_word_ids, token_indice, NGRAM_RANGE)
test_contents = add_ngram(test_word_ids, token_indice, NGRAM_RANGE)

train_padded_seqs = sequence.pad_sequences(train_contents, maxlen=MAX_LENGTH)
test_padded_seqs = sequence.pad_sequences(test_contents, maxlen=MAX_LENGTH)

# Build model
model = FastText(MAX_LENGTH, max_features, EMBEDDING_DIMENSIONS)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

# Fit
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max')

ck_callback = ModelCheckpoint('./checkpoints/fasttext/weights.{epoch:02d}-{val_f1:.4f}.hdf5',
                              monitor='val_f1',
                              mode='max', verbose=2,
                              save_best_only=True,
                              save_weights_only=True)
tb_callback = TensorBoard(log_dir='./logs/fasttext')

model.fit(train_padded_seqs, train_labels,
          batch_size=32, epochs=50,
          callbacks=[early_stopping,
                     Metrics(valid_data=(test_padded_seqs, test_labels)),
                     ck_callback, tb_callback],
          validation_data=(test_padded_seqs, test_labels),
          verbose=0)

model_name = 'fasttext-model-' + JOB_NAME + '.best'
model_dir = os.path.join(base_dir, 'models', 'fasttext', model_name)
model.save(model_dir)
