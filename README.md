# NLP-SentimentSeer

Welcome to **NLP-SentimentSeer**! This repository is dedicated to sentiment classification using various NLP techniques and models. Our mission is to leverage advanced NLP methods to accurately classify sentiment in Twitter data.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Dependencies](#dependencies)
- [Contributors](#contributors)

## Project Overview

This project focuses on sentiment classification using Twitter data. We explore multiple methods including one-hot encoding, TF-IDF, Word2Vec/Glove, and transformer-based models like BERT.

## Dataset

We use a Twitter sentiment dataset containing 1,600,001 rows with sentiment labels (0 for negative, 4 for positive). The dataset is split into training, validation, and test sets.

### Example of the dataset:
- 0 0 1467810369 Mon Apr 06 22:19:45 PDT 2009 NO_QUERY TheSpecialOne @switchfoot http://twitpic.com/2y1zl - Awww, t...
- 1 0 1467810672 Mon Apr 06 22:19:49 PDT 2009 NO_QUERY scotthamilton is upset that he can't update his Facebook by ...
- 2 0 1467810917 Mon Apr 06 22:19:53 PDT 2009 NO_QUERY mattycus @Kenichan I dived many times for the ball. Man...
- 3 0 1467811184 Mon Apr 06 22:19:57 PDT 2009 NO_QUERY ElleCTF my whole body feels itchy and like its on fire
- 4 0 1467811193 Mon Apr 06 22:19:57 PDT 2009 NO_QUERY Karoli @nationwideclass no, it's not behaving at all....


## Preprocessing

### Text Preprocessing
- Convert to lowercase
- Remove punctuation and stopwords
- Lemmatize tokens

### Splitting the Data
The dataset is split into training, validation, and test sets:


### Embeddings
We use one-hot encoding, TF-IDF, Word2Vec, and Glove embeddings to transform the text data into numerical vectors.

## Modeling

### One-Hot Encoding and TF-IDF
1. **Train encoders on training data**
2. **Create embeddings for train, validation, and test sets**
3. **Train classifier with hyperparameter tuning**
4. **Evaluate performance on test data**

### Word2Vec/Glove
1. **Load pre-trained Glove model (glove-twitter-25) or train Word2Vec model**
2. **Create embeddings for train, validation, and test sets**
3. **Train classifier with hyperparameter tuning**
4. **Evaluate performance on test data**

### Transformer (e.g., BERT)
1. **Fine-tune small-scale transformer model for sentiment classification**
2. **Evaluate performance on test data**

## Evaluation

Models are assessed using:
- Accuracy
- Sensitivity
- Specificity
- ROC Curve

## Results

- **One-Hot Encoding and TF-IDF:** Performance metrics based on chosen classifier and tuned hyperparameters.
- **Word2Vec/Glove:** Performance metrics based on chosen classifier and tuned hyperparameters.
- **Transformer Model:** High performance with continuous evaluation across epochs.

## Dependencies
- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- NLTK
- Joblib
- Transformers
- PyTorch

## Contributors
Sruthi Kasturi
