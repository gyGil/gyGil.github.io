---
title: "Machine Translation"
excerpt: ""
last_modified_at:
categories:
  - Natural Language Processing
tags:
  - NLP
  - RNN
  - GRU
  - Keras
  - TensorFlow
---

## Introduction
This project is for translating English sentence to French sentence.
I built a deep neural network(bidirectional RNN with GRU units) as part of end-to-end machine translation pipeline. The completed pipeline will accept English text as input and return the French translation. I compared the results of various RNN structures at the end.

## Process
* Preprocess - Convert text to sequence of integers
* Models - Created bidirectional RNN with GRU units which accepts a sequence of integers as input and returns a probability distribution over possible translations
* Prediction - Run the model on English text.

## Environment
* AWS EC2 p2.xlarge
* Jupyter Notebook
* Python 3.5, Keras, TensorFlow 1.1

## Preprocessing dataset
The most common datasets used for machine translation are from [WMT](https://http://www.statmt.org/). However, that will take a long time to train a neural network on. So, I used comparably the small dataset for my project's purpose.

### *Load Data*
English and French Data are loaded. The both datasets are sequenced already.
```python
import helper

# Load English data
english_sentences = helper.load_data('data/small_vocab_en')
# Load French data
french_sentences = helper.load_data('data/small_vocab_fr')
```
The sample of pair English and French sentence are below.


small_vocab_en Line 1:  *new jersey is sometimes quiet during autumn , and it is snowy in april .*
small_vocab_fr Line 1:  *new jersey est parfois calme pendant l' automne , et il est neigeux en avril .*

small_vocab_en Line 2:  *the united states is usually chilly during july , and it is usually freezing in november .*
small_vocab_fr Line 2: *les états-unis est généralement froid en juillet , et il gèle habituellement en novembre .*

### *Tokenize and Padding*
Tokenize the words into ids.

```python
import project_tests as tests
from keras.preprocessing.text import Tokenizer


def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)

    return tokenizer.texts_to_sequences(x), tokenizer
```

Add padding to make all the sequences the same length.

```python
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    return pad_sequences(x, maxlen=length, padding='post', truncating='post')
```

The below is the results by Tokenizing and Padding.

*The quick brown fox jumps over the lazy dog .*  
[1 2 4 5 6 7 1 8 9]  - Tokenizing  
[1 2 4 5 6 7 1 8 9 0]  - Padding  

## Model

I tested several models to get better accuracy in test dataset. I will present the best model which is bidirectional RNN with GRU and Embedding Layer. RNN with GRU or LSTM is a good neural network model for sequence data like sentence or stock price. Also Embedding layer is one of important concept for Natural Language Processing. At the end, I will compare the results of different models.

### *Embedding Layer*

<img src="https://www.tensorflow.org/images/linear-relationships.png" class="align-center" alt="">  

*Image from _[TensorFlow](https://www.tensorflow.org/tutorials/word2vec)_*      

Word2Vector concept (used in Embedding Layer) is very important in Natural Language Processing. Each word itself which is converted in the number here doesn't have any meaning for machine. So we need to convert the word to the meaningful thing for machine. The word can be converted to the vector using n-gram. The vector presents relations among words. You can check [Embedding Projector](https://projector.tensorflow.org/) of Google visually what it means.

### *Bidirectional RNN*

Bidirectional RNN is
