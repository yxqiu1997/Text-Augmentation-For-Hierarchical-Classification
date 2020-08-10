from keras.layers import Dense, Input, Conv2D, Embedding, Dropout, Reshape, \
                            Flatten, SpatialDropout1D, MaxPool2D, Concatenate, \
                            LSTM, GlobalAveragePooling1D, Bidirectional, \
                            SimpleRNN, Lambda, Conv1D, GlobalMaxPooling1D, \
                            TimeDistributed
from keras.models import Model
from attention import Attention
import numpy as np
import tensorflow as tf


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def get_textcnn_model(maxlen, max_features, embed_size, embedding_matrix):
    filter_sizes = [1, 2, 3, 5]
    num_filters = 32
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.4)(x)
    x = Reshape((maxlen, embed_size, 1))(x)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size), kernel_initializer='normal',
                    activation='relu')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size), kernel_initializer='normal',
                    activation='relu')(x)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), kernel_initializer='normal',
                    activation='relu')(x)
    conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size), kernel_initializer='normal',
                    activation='relu')(x)

    maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1))(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1))(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1))(conv_2)
    maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] + 1, 1))(conv_3)

    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(4, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


class TextRNN(Model):
    def __init__(self, maxlen, max_features, embedding_dims,
                 class_num=4, last_activation='sigmoid'):
        super(TextRNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        self.rnn = LSTM(128)  # LSTM or GRU
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextRNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextRNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        embedding = self.embedding(inputs)
        x = self.rnn(embedding)
        output = self.classifier(x)
        return output


class FastText(Model):
    def __init__(self, maxlen, max_features, embedding_dims,
                 class_num=4, last_activation='sigmoid'):
        super(FastText, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        self.avg_pooling = GlobalAveragePooling1D()
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of FastText must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of FastText must be %d, but now is %d' % (self.maxlen,
                                                                                             inputs.get_shape()[1]))
        embedding = self.embedding(inputs)
        x = self.avg_pooling(embedding)
        output = self.classifier(x)
        return output


class TextBiRNN(Model):
    def __init__(self, maxlen, max_features, embedding_dims,
                 class_num=4, last_activation='sigmoid'):
        super(TextBiRNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        self.bi_rnn = Bidirectional(LSTM(128))  # LSTM or GRU
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextBiRNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextBiRNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        embedding = self.embedding(inputs)
        x = self.bi_rnn(embedding)
        output = self.classifier(x)
        return output


class TextAttBiRNN(Model):
    def __init__(self, maxlen, max_features, embedding_dims,
                 class_num=4, last_activation='sigmoid'):
        super(TextAttBiRNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        self.bi_rnn = Bidirectional(LSTM(128, return_sequences=True))  # LSTM or GRU
        self.attention = Attention(self.maxlen)
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextAttBiRNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextAttBiRNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        embedding = self.embedding(inputs)
        x = self.bi_rnn(embedding)
        x = self.attention(x)
        output = self.classifier(x)
        return output


class RCNN(Model):
    def __init__(self, maxlen, max_features,embedding_dims,
                 class_num=4, last_activation='sigmoid'):
        super(RCNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        self.forward_rnn = SimpleRNN(128, return_sequences=True)
        self.backward_rnn = SimpleRNN(128, return_sequences=True, go_backwards=True)
        self.reverse = Lambda(lambda x: tf.reverse(x, axis=[1]))
        self.concatenate = Concatenate(axis=2)
        self.conv = Conv1D(64, kernel_size=1, activation='tanh')
        self.max_pooling = GlobalMaxPooling1D()
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs) != 3:
            raise ValueError('The length of inputs of RCNN must be 3, but now is %d' % len(inputs))
        input_current = inputs[0]
        input_left = inputs[1]
        input_right = inputs[2]
        if len(input_current.get_shape()) != 2 or len(input_left.get_shape()) != 2 or len(input_right.get_shape()) != 2:
            raise ValueError('The rank of inputs of RCNN must be (2, 2, 2), but now is (%d, %d, %d)' % (len(input_current.get_shape()), len(input_left.get_shape()), len(input_right.get_shape())))
        if input_current.get_shape()[1] != self.maxlen or input_left.get_shape()[1] != self.maxlen or input_right.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of RCNN must be (%d, %d, %d), but now is (%d, %d, %d)' % (self.maxlen, self.maxlen, self.maxlen, input_current.get_shape()[1], input_left.get_shape()[1], input_right.get_shape()[1]))
        embedding_current = self.embedding(input_current)
        embedding_left = self.embedding(input_left)
        embedding_right = self.embedding(input_right)
        x_left = self.forward_rnn(embedding_left)
        x_right = self.backward_rnn(embedding_right)
        x_right = self.reverse(x_right)
        x = self.concatenate([x_left, embedding_current, x_right])
        x = self.conv(x)
        x = self.max_pooling(x)
        output = self.classifier(x)
        return output


class RCNNVariant(Model):
    """Variant of RCNN.
        Base on structure of RCNN, we do some improvement:
        1. Ignore the shift for left/right context.
        2. Use Bidirectional LSTM/GRU to encode context.
        3. Use Multi-CNN to represent the semantic vectors.
        4. Use ReLU instead of Tanh.
        5. Use both AveragePooling and MaxPooling.
    """

    def __init__(self,maxlen, max_features, embedding_dims,
                 kernel_sizes=[1, 2, 3, 4, 5],
                 class_num=4, last_activation='sigmoid'):
        super(RCNNVariant, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.kernel_sizes = kernel_sizes
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        self.bi_rnn = Bidirectional(LSTM(128, return_sequences=True))
        self.concatenate = Concatenate()
        self.convs = []
        for kernel_size in self.kernel_sizes:
            conv = Conv1D(128, kernel_size, activation='relu')
            self.convs.append(conv)
        self.avg_pooling = GlobalAveragePooling1D()
        self.max_pooling = GlobalMaxPooling1D()
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextRNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextRNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        embedding = self.embedding(inputs)
        x_context = self.bi_rnn(embedding)
        x = self.concatenate([embedding, x_context])
        convs = []
        for i in range(len(self.kernel_sizes)):
            conv = self.convs[i](x)
            convs.append(conv)
        poolings = [self.avg_pooling(conv) for conv in convs] + [self.max_pooling(conv) for conv in convs]
        x = self.concatenate(poolings)
        output = self.classifier(x)
        return output


class HAN(Model):
    def __init__(self, maxlen_sentence, maxlen_word,
                 max_features, embedding_dims,
                 class_num=4, last_activation='sigmoid'):
        super(HAN, self).__init__()
        self.maxlen_sentence = maxlen_sentence
        self.maxlen_word = maxlen_word
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        # Word part
        input_word = Input(shape=(self.maxlen_word,))
        x_word = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen_word)(input_word)
        x_word = Bidirectional(LSTM(128, return_sequences=True))(x_word)  # LSTM or GRU
        x_word = Attention(self.maxlen_word)(x_word)
        model_word = Model(input_word, x_word)
        # Sentence part
        self.word_encoder_att = TimeDistributed(model_word)
        self.sentence_encoder = Bidirectional(LSTM(128, return_sequences=True))  # LSTM or GRU
        self.sentence_att = Attention(self.maxlen_sentence)
        # Output part
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 3:
            raise ValueError('The rank of inputs of HAN must be 3, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen_sentence:
            raise ValueError('The maxlen_sentence of inputs of HAN must be %d, but now is %d' % (self.maxlen_sentence, inputs.get_shape()[1]))
        if inputs.get_shape()[2] != self.maxlen_word:
            raise ValueError('The maxlen_word of inputs of HAN must be %d, but now is %d' % (self.maxlen_word, inputs.get_shape()[2]))
        x_sentence = self.word_encoder_att(inputs)
        x_sentence = self.sentence_encoder(x_sentence)
        x_sentence = self.sentence_att(x_sentence)
        output = self.classifier(x_sentence)
        return output
