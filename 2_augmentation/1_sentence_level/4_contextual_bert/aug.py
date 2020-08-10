import os
import nlpaug.augmenter.word as naw
from absl import app
import sys


base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, 'data')
model_dir = os.path.join(base_dir, 'model')
raw_dir = os.path.join(data_dir, 'raw')
aug_dir = os.path.join(data_dir, 'aug')


def main(_):
    aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', aug_p=0.1)
    index = sys.argv[1]

    train_filename = 'train-' + str(index) + '.txt'
    aug_filename = 'aug-' + str(index) + '.txt'
    train_file = os.path.join(raw_dir, train_filename)
    aug_file = os.path.join(aug_dir, aug_filename)

    with open(train_file, 'r', encoding='utf-8') as r:
        with open(aug_file, 'w', encoding='utf-8') as w:
            lines = r.readlines()
            cnt = 0
            for line in lines:
                w.write(aug.augment(line.strip()) + '\n')
                cnt += 1
                print('********' + str(index) + '--' + str(cnt) + '********')


if __name__ == '__main__':
    app.run(main)
