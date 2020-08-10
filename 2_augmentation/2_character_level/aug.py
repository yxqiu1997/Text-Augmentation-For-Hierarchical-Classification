from absl import app
import nlpaug.augmenter.char as nac
import os


def main(_):
    base_dir = os.path.dirname(__file__)
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    aug_dir = os.path.join(base_dir, 'data', 'aug')
    delete_dir = os.path.join(aug_dir, 'delete')
    insert_dir = os.path.join(aug_dir, 'insert')
    keyboard_dir = os.path.join(aug_dir, 'keyboard')
    ocr_dir = os.path.join(aug_dir, 'ocr')
    substitute_dir = os.path.join(aug_dir, 'substitute')
    swap_dir = os.path.join(aug_dir, 'swap')

    for index in range(0, 12):
        raw_filename = 'train-' + str(index) + '.txt'
        raw_file = os.path.join(raw_dir, raw_filename)
        with open(raw_file, 'r', encoding='utf-8') as r:
            lines = r.readlines()
            for line in lines:
                text = line.strip()

                # Substitute character by pre-defined OCR error
                ocr_filename = 'ocr-' + str(index) + '.txt'
                ocr_file = os.path.join(ocr_dir, ocr_filename)
                aug = nac.OcrAug()
                augmented_texts = aug.augment(text, n=1)
                with open(ocr_file, 'a', encoding='utf-8') as w:
                    w.write(augmented_texts + '\n')

                # Substitute character by keyboard distance
                keyboard_filename = 'keyboard-' + str(index) + '.txt'
                keyboard_file = os.path.join(keyboard_dir, keyboard_filename)
                aug = nac.KeyboardAug()
                augmented_texts = aug.augment(text)
                with open(keyboard_file, 'a', encoding='utf-8') as w:
                    w.write(augmented_texts + '\n')

                # Insert character randomly
                insert_filename = 'insert-' + str(index) + '.txt'
                insert_file = os.path.join(insert_dir, insert_filename)
                aug = nac.RandomCharAug(action='insert')
                augmented_texts = aug.augment(text)
                with open(insert_file, 'a', encoding='utf-8') as w:
                    w.write(augmented_texts + '\n')

                # Substitute character randomly
                substitute_filename = 'substitute-' + str(index) + '.txt'
                substitute_file = os.path.join(substitute_dir, substitute_filename)
                aug = nac.RandomCharAug(action='substitute')
                augmented_texts = aug.augment(text)
                with open(substitute_file, 'a', encoding='utf-8') as w:
                    w.write(augmented_texts + '\n')

                # Swap character randomly
                swap_filename = 'swap-' + str(index) + '.txt'
                swap_file = os.path.join(swap_dir, swap_filename)
                aug = nac.RandomCharAug(action='swap')
                augmented_texts = aug.augment(text)
                with open(swap_file, 'a', encoding='utf-8') as w:
                    w.write(augmented_texts + '\n')

                # Delete character randomly
                delete_filename = 'delete-' + str(index) + '.txt'
                delete_file = os.path.join(delete_dir, delete_filename)
                aug = nac.RandomCharAug(action='delete')
                augmented_texts = aug.augment(text)
                with open(delete_file, 'a', encoding='utf-8') as w:
                    w.write(augmented_texts + '\n')

        print('Finished train-' + str(index) + '.txt')


if __name__ == "__main__":
    app.run(main)
