from transformers import MarianMTModel, MarianTokenizer
import os
import sys
from absl import app


def main(_):
    base_dir = os.path.dirname(__file__)
    model_name = os.path.join(base_dir, 'models', 'en-de')

    index = sys.argv[1]
    en_file = 'train-' + str(index) + '.txt'
    en_dir = os.path.join(base_dir, 'data', 'raw', 'content', en_file)

    de_file = 'de-' + str(index) + '.txt'
    de_dir = os.path.join(base_dir, 'data', 'de', de_file)

    with open(en_dir, 'r', encoding='utf-8') as r:
        with open(de_dir, 'w', encoding='utf-8') as w:
            en_list = r.readlines()
            for line in en_list:
                src = '>>de<< ' + line
                src_text = [src]

                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                translated = model.generate(**tokenizer.prepare_translation_batch(src_text))

                tgt = tokenizer.decode(translated[0], skip_special_tokens=True)
                w.write(tgt + '\n')


if __name__ == "__main__":
    app.run(main)
