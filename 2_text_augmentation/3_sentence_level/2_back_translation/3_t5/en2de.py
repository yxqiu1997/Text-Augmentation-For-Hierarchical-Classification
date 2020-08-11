from transformers import T5Tokenizer, T5ForConditionalGeneration
from absl import app
import os
import sys


tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')
index = sys.argv[1]


def main(_):
    base_dir = os.path.dirname(__file__)
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    de_dir = os.path.join(base_dir, 'data', 'de')
    prefix = 'translate English to German: '

    en_filename = 'train-' + str(index) + '.txt'
    en_dir = os.path.join(raw_dir, en_filename)
    de_filename = 'de-' + str(index) + '.txt'
    de_dir = os.path.join(de_dir, de_filename)

    with open(en_dir, 'r', encoding='utf-8') as r:
        with open(de_dir, 'w', encoding='utf-8') as w:
            lines = r.readlines()
            for line in lines:
                text = prefix + line
                input_ids = tokenizer.encode(text, return_tensors='pt')
                outputs = model.generate(input_ids)
                de_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                w.write(de_text + '\n')


if __name__ == '__main__':
    app.run(main)
