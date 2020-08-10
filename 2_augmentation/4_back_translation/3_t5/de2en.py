from transformers import MarianMTModel, MarianTokenizer
import os
import sys
from absl import app


def main(_):
    base_dir = os.path.dirname(__file__)
    model_name = os.path.join(base_dir, 'models', 'de-en')

    index = sys.argv[1]
    de_file = 'de-' + str(index) + '.txt'
    de_dir = os.path.join(base_dir, 'data', 'de', de_file)

    aug_file = 'aug-' + str(index) + '.txt'
    aug_dir = os.path.join(base_dir, 'data', 'aug', aug_file)

    with open(de_dir, 'r', encoding='utf-8') as r:
        with open(aug_dir, 'w', encoding='utf-8') as w:
            de_list = r.readlines()
            for line in de_list:
                src = '>>en<< ' + line
                src_text = [src]

                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                translated = model.generate(**tokenizer.prepare_translation_batch(src_text))

                tgt = tokenizer.decode(translated[0], skip_special_tokens=True)
                w.write(tgt + '\n')


if __name__ == "__main__":
    app.run(main)
