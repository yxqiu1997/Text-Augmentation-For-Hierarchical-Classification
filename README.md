# Brief Introduction

<a name="_overview"></a>

## Navigation
The whole project can be divided into four parts:

* [Cleaning and preprocessing of raw data](1_data_preprocessing/README.md)
* [Text augmentation](2_text_augmentation/README.md)
* [Hierarchical classification](3_hierarchical_classification/README.md)
* [Evaluation of augmentation quality](4_quality_evaluation/README.md)


<a name="_note"></a>

## Note
Due to the limitation of the file uploading policy of Github, necessary files whose sizes are over 100 Mb have to be ignored. If the project folder holds either the 0_download.sh or the 0_combine_sub_train_content.py script, run them first to get a completed experimental environment.


<a name="_best_result"></a>

## Best Result

### Augmentation Technique

The table below lists the highest Macro F1-score (based on the classifier performances) achieved by different augmentation techniques: 

| Classifier | Macro F1-score | Augmentation Technique |
| :----: | :----: | :----: | 
| Fasttext | 0.91158 | Keyboard Error |
| CNN | 0.91145 | Keyboard Error |
| RNN | 0.90711 | T5 Model |
| Bi-RNN | 0.90737 | T5 Model |
| Attention-Bi-RNN | 0.90750 | Word Deletion |
| RCNN | 0.90816 | T5 Model |
| HAN | 0.90750 | T5 Model & Word Deletion |

Detailed results can be seen at: 


### Classifier Performance
The table below lists the most suitable classifier(s) for different augmentation techniques (based on Macro F1-score).

Without augmentation:

Augmentation Technique | Classifier | Macro F1-score |
| :----: | :----: | :----: | 
| NULL | CNN | 0.90789 |


Augmented at character level:

Augmentation Technique | Classifier | Macro F1-score |
| :----: | :----: | :----: | 
| Character Deletion | HAN | 0.90750 |
| Character Insertion | Fasttext Model | 0.91066 |
| Keyboard Error | Fasttext Model | 0.91158 |
| OCR Error | CNN | 0.90724 |
| Character Substitution | Fasttext Model | 0.91145 |
| Character Swap | CNN | 0.90750 |


Augmented at word level:

Augmentation Technique | Classifier | Macro F1-score |
| :----: | :----: | :----: | 
| Word Deletion | CNN | 0.90816 |
| Word Insertion | CNN | 0.90803 |
| Word Swap | CNN | 0.90803 |
| Synonyms Replacement | HAN | 0.90842 |
| TF-IDF | Attention-Bi-RNN | 0.90421 |
| BERT Model | CNN | 0.90921 |
| DistilBERT Model | Fasttext Model | 0.90658 |
| RoBERTa Model | Bi-RNN | 0.90263 |
| Contextual BERT Model | CNN | 0.90684 |


Augmented at sentence level:
Augmentation Technique | Classifier | Macro F1-score |
| :----: | :----: | :----: | 
| GPT-2 Model | Fasttext Model | 0.90461 |
| XLNet Model | Fasttext Model & Bi-RNN | 0.90618 |
| Sequence2Sequence Model | Fasttext Model | 0.91000 |
| Tranformer Model | CNN | 0.90816 |
| T5 Model | CNN | 0.90876 |

