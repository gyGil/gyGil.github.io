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

#### *Load Data*
English and French Data are loaded. The both datasets are sequenced already.
```python
import helper

# Load English data
english_sentences = helper.load_data('data/small_vocab_en')
# Load French data
french_sentences = helper.load_data('data/small_vocab_fr')
```
The sample of pair English and French sentence are below.

```
small_vocab_en Line 1:  new jersey is sometimes quiet during autumn , and it is snowy in april.
small_vocab_fr Line 1:  new jersey est parfois calme pendant l' automne , et il est neigeux en avril.

small_vocab_en Line 2:  the united states is usually chilly during july , and it is usually freezing in november.
small_vocab_fr Line 2:  les états-unis est généralement froid en juillet , et il gèle habituellement en novembre.
```
