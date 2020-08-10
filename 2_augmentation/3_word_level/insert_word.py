from absl import app
import os
import nlpaug.augmenter.word as naw
import sys


base_dir = os.path.dirname(__file__)
raw_dir = os.path.join(base_dir, 'data', 'raw', 'content')
aug_dir = os.path.join(base_dir, 'data', 'aug_word')


def main(_):
    index = sys.argv[1]

    # Insert word by BERT
    aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased',
                                    action='insert')

    train_filename = 'train-' + str(index) + '.txt'
    aug_filename = 'aug-' + str(index) + '.txt'
    train_file = os.path.join(raw_dir, train_filename)
    aug_file = os.path.join(aug_dir, 'BERT', aug_filename)

    with open(train_file, 'r', encoding='utf-8') as r:
        with open(aug_file, 'w', encoding='utf-8') as w:
            lines = r.readlines()
            cnt = 1
            for line in lines:
                w.write(aug.augment(line.strip()) + '\n')
                print('***** 1/3 - ' + str(index) + ' - ' + str(cnt) + ' *****')
                cnt += 1

    # Insert word by DistilBERT
    aug = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased',
                                        action='insert')

    train_filename = 'train-' + str(index) + '.txt'
    aug_filename = 'aug-' + str(index) + '.txt'
    train_file = os.path.join(raw_dir, train_filename)
    aug_file = os.path.join(aug_dir, 'DistilBERT', aug_filename)

    with open(train_file, 'r', encoding='utf-8') as r:
        with open(aug_file, 'w', encoding='utf-8') as w:
            lines = r.readlines()
            cnt = 1
            for line in lines:
                w.write(aug.augment(line.strip()) + '\n')
                print('***** 2/3 - ' + str(index) + ' - ' + str(cnt) + ' *****')
                cnt += 1

    # Insert word by RoBERTA
    aug = naw.ContextualWordEmbsAug(model_path='roberta-base',
                                    action='insert')

    train_filename = 'train-' + str(index) + '.txt'
    aug_filename = 'aug-' + str(index) + '.txt'
    train_file = os.path.join(raw_dir, train_filename)
    aug_file = os.path.join(aug_dir, 'RoBERTA', aug_filename)

    with open(train_file, 'r', encoding='utf-8') as r:
        with open(aug_file, 'w', encoding='utf-8') as w:
            lines = r.readlines()
            cnt = 1
            for line in lines:
                w.write(aug.augment(line.strip()) + '\n')
                print('***** 3/3 - ' + str(index) + ' - ' + str(cnt) + ' *****')
                cnt += 1


if __name__ == '__main__':
    app.run(main)
