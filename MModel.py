#!/usr/bin/env python
# coding: utf-8

# In[1]:

#Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import arabic_reshaper
from bidi.algorithm import get_display
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

rand_seed = 0  # random state for reproducibility
np.random.seed(rand_seed)


# In[2]:

#Function to split the dataset into training and testing data based on a fraction
def random_split(data, features, output, fraction, seed=0):
    X_train, X_test, y_train, y_test = train_test_split(data[features],
                                                        data[output],
                                                        stratify = data[output],
                                                        random_state=seed,
                                                        train_size=fraction
                                                       )
    train_data = pd.DataFrame(data=X_train, columns=features)
    train_data[output] = y_train
    test_data = pd.DataFrame(data=X_test, columns=features)
    test_data[output] = y_test
    
    return train_data, test_data


# In[3]:

#Function to run the classifier and output the score on both training and testing data
def train_test_classifier(classifier, train_features, train_labels, test_features, test_labels):
    classifier.fit(train_features, train_labels)

    print("Score on Training Data:")
    print(classifier.score(train_features, train_labels))
    print('_'*100)
    
    pred_y = classifier.predict(test_features)
    print("Score on Testing Data:")
    print(accuracy_score(test_labels, pred_y))
    


# In[4]:

#Function to tokenize the tweets
def get_words(text):
    tokens = word_tokenize(text)  # Split text into words
    return tokens


# In[5]:

#Reading the normalized excel file
data = pd.read_excel("balanced_medium.xlsx")
data = data.dropna() #Dropping empty rows
data.head()


# In[6]:

#Finding out the number of tweets for each sentiment 
pos = data[data['Sentiment'] == 'POS']
neg = data[data['Sentiment'] == 'NEG']
neu = data[data['Sentiment'] == 'NEUTRAL']
obj = data[data['Sentiment'] == 'OBJ']
print(f'The number of Positive Tweets is:  {len(pos)}')
print(f'The number of Negative Tweets is:  {len(neg)}')
print(f'The number of Neutral Tweets is:  {len(neu)}')
print(f'The number of Objective Tweets is:  {len(obj)}')


# In[7]:

#Creating a Pie Chart to represent the distribution of tweets in relation to sentiments
labels = "Positive", "Negative", "Neutral", "Objective"
sizes = [len(pos), len(neg), len(neu), len(obj)]

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels)
plt.title("Distribution of Sentiments in Dataset")
plt.tight_layout()


# In[8]:

#setting the training dataset to 80% of the dataset
train_fraction = .80 

output = 'Sentiment'
features = data.columns.tolist() # the features columns
features.remove(output)
print('Output:', output)
print('Features:', features)
train_data, test_data = random_split(data, features, output, train_fraction, rand_seed)

print(f"Total Tweets of this Dataset: {len(data)}")
print(f"Total Training Tweets of this Dataset: {len(train_data)}")
print(f"Total Testing Tweets of this Dataset: {len(test_data)}")


# In[9]:


train_data.head()


# In[10]:

#Using TF-IDF to get the training and testing features
vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_df=0.5, stop_words=None, use_idf=True)

train_data_features = vectorizer.fit_transform(train_data['Text'].values.astype('U'))
test_data_features = vectorizer.transform(test_data['Text'].values.astype('U'))


# In[11]:

#Grouping the words for each sentiment
all_positive_words = pos["Text"].apply(get_words)
all_negative_words = neg["Text"].apply(get_words)
all_neutral_words = neu["Text"].apply(get_words)
all_objective_words = obj["Text"].apply(get_words)


# In[12]:

#Flattening the words for each sentiment to find out their frequency distribution
all_positive_words_flat = [word for sublist in all_positive_words for word in sublist]
all_negative_words_flat = [word for sublist in all_negative_words for word in sublist]
all_neutral_words_flat = [word for sublist in all_neutral_words for word in sublist]
all_objective_words_flat = [word for sublist in all_objective_words for word in sublist]

positive_frequency_distribution = nltk.FreqDist(all_positive_words_flat)
negative_frequency_distribution = nltk.FreqDist(all_negative_words_flat)
neutral_frequency_distribution = nltk.FreqDist(all_neutral_words_flat)
objective_frequency_distribution = nltk.FreqDist(all_objective_words_flat)


# In[13]:

#Sorting and reshaping of positive words
sorted_positive_words = sorted(positive_frequency_distribution, key=positive_frequency_distribution.get, reverse=True)
positive_words = sorted_positive_words[:15]
positive_counts = [positive_frequency_distribution[word] for word in positive_words]

reshaped_positive_words = [arabic_reshaper.reshape(word) for word in positive_words]
display_positive_words = [get_display(word) for word in reshaped_positive_words]

#Horizantal bar chart to display the 15 most frequently used positive words
fig, ax = plt.subplots(figsize=(15, 10))
plt.barh(display_positive_words[::-1], positive_counts[::-1], color='#0cbbf5ff')
plt.title('Top words in positive tweets', fontsize=18)
plt.xlabel('Count', fontsize=16)
plt.ylabel('Words', fontsize=16)
plt.tight_layout()
plt.show()


# In[14]:

#Sorting and reshaping of negative words
sorted_negative_words = sorted(negative_frequency_distribution, key=negative_frequency_distribution.get, reverse=True)
negative_words = sorted_negative_words[:15]
negative_counts = [negative_frequency_distribution[word] for word in negative_words]

reshaped_negative_words = [arabic_reshaper.reshape(word) for word in negative_words]
display_negative_words = [get_display(word) for word in reshaped_negative_words]

#Horizantal bar chart to display the 15 most frequently used negative words
fig, ax = plt.subplots(figsize=(15, 10))
plt.barh(display_negative_words[::-1], negative_counts[::-1], color='#0cbbf5ff')
plt.title('Top words in negative tweets', fontsize=18)
plt.xlabel('Count', fontsize=16)
plt.ylabel('Words', fontsize=16)
plt.tight_layout()
plt.show()


# In[15]:

#Sorting and reshaping of neutral words
sorted_neutral_words = sorted(neutral_frequency_distribution, key=neutral_frequency_distribution.get, reverse=True)
neutral_words = sorted_neutral_words[:15]
neutral_counts = [neutral_frequency_distribution[word] for word in neutral_words]

reshaped_neutral_words = [arabic_reshaper.reshape(word) for word in neutral_words]
display_neutral_words = [get_display(word) for word in reshaped_neutral_words]

#Horizantal bar chart to display the 15 most frequently used neutral words
fig, ax = plt.subplots(figsize=(15, 10))
plt.barh(display_neutral_words[::-1], neutral_counts[::-1], color='#0cbbf5ff')
plt.title('Top words in neutral tweets', fontsize=18)
plt.xlabel('Count', fontsize=16)
plt.ylabel('Words', fontsize=16)
plt.tight_layout()
plt.show()


# In[16]:

#Sorting and reshaping of objective words
sorted_objective_words = sorted(objective_frequency_distribution, key=objective_frequency_distribution.get, reverse=True)
objective_words = sorted_objective_words[:15]
objective_counts = [objective_frequency_distribution[word] for word in objective_words]

reshaped_objective_words = [arabic_reshaper.reshape(word) for word in objective_words]
display_objective_words = [get_display(word) for word in reshaped_objective_words]

#Horizantal bar chart to display the 15 most frequently used objective words
fig, ax = plt.subplots(figsize=(15, 10))
plt.barh(display_objective_words[::-1], objective_counts[::-1], color='#0cbbf5ff')
plt.title('Top words in objective tweets', fontsize=18)
plt.xlabel('Count', fontsize=16)
plt.ylabel('Words', fontsize=16)
plt.tight_layout()
plt.show()


# In[17]:

#Getting the number of tweets and number of words in both training and testing datasets
train_data_features.shape, test_data_features.shape


# In[18]:

#Applying the different classifiers
logistic_reg = LogisticRegression(random_state=rand_seed)

train_test_classifier(logistic_reg, train_data_features, train_data[output],
                        test_data_features, test_data[output])


# In[19]:


mnb = MultinomialNB()

train_test_classifier(mnb, train_data_features, train_data[output],
                        test_data_features, test_data[output])


# In[20]:


svm = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)

train_test_classifier(svm, train_data_features, train_data[output],
                        test_data_features, test_data[output])


# In[21]:


rf = RandomForestClassifier(n_estimators=100, random_state=rand_seed)

train_test_classifier(rf, train_data_features, train_data[output],
                        test_data_features, test_data[output])


# In[22]:


mlp = MLPClassifier(hidden_layer_sizes=(20,20,20,20), verbose=True, tol=0.001, random_state=rand_seed)
train_test_classifier(mlp, train_data_features, train_data[output],
                        test_data_features, test_data[output])


