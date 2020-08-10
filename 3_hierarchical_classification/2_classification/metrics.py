from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import tensorflow as tf
import numpy as np


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')
        _val_accuracy = accuracy_score(val_targ, val_predict)

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        logs['val_accuracy'] = _val_accuracy
        print()
        print('***** macro: *****')
        print(" — f1: %f — precision: %f — recall: %f - accuracy: %f"
              % (_val_f1, _val_precision, _val_recall, _val_accuracy))

        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')
        _val_accuracy = accuracy_score(val_targ, val_predict)

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        logs['val_accuracy'] = _val_accuracy
        print()
        print('***** micro: *****')
        print(" — f1: %f — precision: %f — recall: %f - accuracy: %f"
              % (_val_f1, _val_precision, _val_recall, _val_accuracy))

        return
