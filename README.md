# Brief Introduction

<a name="_overview"></a>

## Navigation
The whole project can be divided into four parts:

* Cleaning and preprocessing of raw data
* [Text augmentation](#_best_augment)
* [Hierarchical classification](#_best_classify)
* [Evaluation of augmentation quality](#_best_evaluate)


<a name="_note"></a>

## Note
Due to the limitation of the file uploading policy of Github, necessary files whose sizes are over 100 Mb have to be ignored. If the project folder holds either the 0_download.sh or the 0_combine_sub_train_content.py script, run them first to get a completed experimental environment.


<a name="_best_result"></a>

## Best Result

<a name="_best_augment"></a>

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


<a name="_best_classify"></a>

### Classifier Performance
The table below lists the most suitable classifier(s) for different augmentation techniques (based on Macro F1-score).

Without augmentation:

Augmentation Technique | Classifier | Macro F1-score |
| :----: | :----: | :----: | 
| N/A | CNN | 0.90789 |


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
| Transformer Model | CNN | 0.90816 |
| T5 Model | CNN | 0.90876 |



<a name="_best_evaluate"></a>

### Back-translation Quality

No golden English-German and German-English references are provided in this project because of the lack of human-language experts. The whole evaluation process has to rely on the feature of back translation technique. In other words, the back-translated document must be compared with the source document. Suppose there is a workflow like this: English file A --> German file B --> English file C, then the file C will be evaluated in the case where the file A is the reference. This limitation inevitably leads to low scores on traditional NMT evaluation metrics. 

| Translation Technique | BLEU | METEOR | Rouge-L | CIDEr |
| :----: | :----: | :----: | :----: | :----: | 
| Sequence2Sequence Model | 0.1273 | 0.2114 | 0.2703 | 1.0208 |
| T5 Model | 0.1613 | 0.2164 | 0.3850 | 1.0552 |
| Transformer Model | 0.5552 | 0.4967 | 0.5765 | 3.3353 | 


To mitigate the advert impact, a new model (Pre-trained BERT + Bi-LSTM) is proposed to evaluate the translation quality. There are two stages in this part of work. Firstly, evaluate this new model on the data set of WMT17 and compare it with baseline model (QuEst++). Pearson correlation coefficient is used as the main metric. 
<!-- Pearson correlation coefficient, mean absolute error and mean squared error -->

| NMT Evaluation Model | Pearson's r | MAE | RMSE |
| :----: | :----: | :----: | :----: |
| QuEst++ | 0.397 | 0.136 | 0.175 |
| BERT + Bi-LSTM | N/A | N/A | N/A |


Secondly, feed texts augmented by back-translation techniques in this project into the new model and get the final score:

| Translation Technique | Score |
| :----: | :----: |
| Sequence2Sequence Model | N/A |
| T5 Model | N/A |
| Transformer Model | N/A |

