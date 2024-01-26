# Arabic-Tweets-Sentiment-Analysis
This repository covers the different stages that I went through to complete a sentiment analysis model of Arabic Tweets using NLP. In addition, it goes through the normalization process of the data, the implementation of the model with different datasets, and the results that different classifiers achieved with the created model.

# Normalization
Text normalization is one of the most important steps in NLP Programs. By normalizing text, we are able to reduce its randomness and thereby deduce more accurate insights. It also reduces the amount of information that the computer has to deal with.

Naturally, normalizing Arabic text is way different than normalizing English text. To normalize Arabic text, I need to take into account many considerations. First, I need to remove punctuation, non-Arabic words, and extra spaces. I also need to remove Tatweel, Tashkeel, and the Arabic diacritics. I also need to take care of converting eastern Arabic numbers to western Arabic numbers, and to normalize the spell errors. Finally, I need to remove emails, URLs, and phone numbers.
First, I started my implementation of the normalization file by importing the libraries I want to use. These libraries are re, string, nltk.corpus package, pandas, and itertools.

I went through different steps to clean the data including unicode identification, removings tashkeel, tatweel, stop words, and more.

# Models
I created 3 models, each trained on a dataset with a different size. Due to the datasets not being usable, the SModel and MModel didn't perform well. This can be seen by looking at the features of the datasets and doing some analysis on them. 
