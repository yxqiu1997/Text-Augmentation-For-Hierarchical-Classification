import os
import nagisa
import nlpaug.augmenter.word as naw
from absl import app
# from nlpaug.util.file.download import DownloadUtil
# DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir='models/')


def main(_):
    for index in range(0, 12):
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, 'data')
        train_file = 'train-' + str(index) + '.txt'
        train_dir = os.path.join(data_dir, 'raw', train_file)

        aug_file = 'aug-' + str(index) + '.txt'
        aug_dir = os.path.join(data_dir, 'aug', aug_file)

        with open(train_dir, 'r', encoding='utf-8') as r:
            with open(aug_dir, 'w', encoding='utf-8') as w:
                lines = r.readlines()
                for line in lines:
                    text = line.strip()

                    def tokenizer(x):
                        return nagisa.tagging(text).words

                    aug = naw.WordEmbsAug(model_type='fasttext', tokenizer=tokenizer,
                                          model_path=os.path.join('models', 'wiki-news-300d-1M.vec'))
                    augmented_text = aug.augment(text)
                    w.write(augmented_text + '\n')


if __name__ == "__main__":
    app.run(main)
