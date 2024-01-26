#!/usr/bin/env python
# coding: utf-8

# In[1]:

#Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import arabic_reshaper
import emojis
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

    print("Score on Testing Data:")
    
    pred_y = classifier.predict(test_features)
    print('Accuracy_score: ')
    print(f'{accuracy_score(test_labels, pred_y)*100}%')


# In[4]:

#Function to tokenize the tweets
def get_words(text):
    tokens = word_tokenize(text)  # Split text into words
    return tokens


# In[5]:

#Reading the normalized excel file
data = pd.read_excel("balanced_large.xlsx")
data = data.dropna() #Dropping empty rows
data.head()


# In[6]:

#Finding out the number of tweets for each sentiment 
pos = data[data['Sentiment'] == 'pos']
neg = data[data['Sentiment'] == 'neg']
print(f'The number of Positive Tweets is:  {len(pos)}')
print(f'The number of Negative Tweets is:  {len(neg)}')


# In[7]:

#Creating a Pie Chart to represent the distribution of tweets in relation to sentiments
labels = "Positive", "Negative"
sizes = [len(pos), len(neg)]

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
print('output:', output)
print('features:', features)
train_data, test_data = random_split(data, features, output, train_fraction, rand_seed)

print(len(train_data))

print(len(test_data))
print(len(train_data)+len(test_data))
print(len(data))


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


# In[12]:

#Flattening the words for each sentiment to find out their frequency distribution
all_positive_words_flat = [word for sublist in all_positive_words for word in sublist]
all_negative_words_flat = [word for sublist in all_negative_words for word in sublist]


positive_frequency_distribution = nltk.FreqDist(all_positive_words_flat)
negative_frequency_distribution = nltk.FreqDist(all_negative_words_flat)


# In[13]:

#Sorting and reshaping of positive words
sorted_positive_words = sorted(positive_frequency_distribution, key=positive_frequency_distribution.get, reverse=True)
positive_words = sorted_positive_words[:15]
positive_counts = [positive_frequency_distribution[word] for word in positive_words]

reshaped_positive_words = [arabic_reshaper.reshape(word) for word in positive_words]
display_positive_words = [get_display(word) for word in reshaped_positive_words]

#Horizantal bar chart to display the 15 most frequently used positive words
fig, ax = plt.subplots(figsize=(15, 10))
emojis_list = [emojis.decode(word) for word in display_positive_words[::-1]]

plt.barh(emojis_list, positive_counts[::-1], color='#0cbbf5ff')
plt.title('Top Words in Positive Tweets', fontsize=18)
plt.xlabel('Count', fontsize=16)
plt.ylabel('Words', fontsize=16)
plt.show()


# In[14]:

#Sorting and reshaping of negative words
sorted_negative_words = sorted(negative_frequency_distribution, key=negative_frequency_distribution.get, reverse=True)
negative_words = sorted_negative_words[:15]
negative_counts = [negative_frequency_distribution[word] for word in negative_words]

reshaped_negative_words = [arabic_reshaper.reshape(word) for word in negative_words]
display_negative_words = [get_display(word) for word in reshaped_negative_words]

fig, ax = plt.subplots(figsize=(15, 10))
emojis_list = [emojis.decode(word) for word in display_negative_words[::-1]]

#Horizantal bar chart to display the 15 most frequently used negative words
plt.barh(emojis_list, negative_counts[::-1], color='#0cbbf5ff')
plt.title('Top Words in Negative Tweets', fontsize=18)
plt.xlabel('Count', fontsize=16)
plt.ylabel('Words', fontsize=16)
plt.tight_layout()
plt.show()


# In[15]:

#Getting the number of tweets and number of words in both training and testing datasets
train_data_features.shape, test_data_features.shape


# In[16]:

#Applying the different classifiers
logistic_reg = LogisticRegression(random_state=rand_seed)

train_test_classifier(logistic_reg, train_data_features, train_data[output],
                        test_data_features, test_data[output])


# In[17]:


mnb = MultinomialNB()

train_test_classifier(mnb, train_data_features, train_data[output],
                        test_data_features, test_data[output])


# In[18]:


svm = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)

train_test_classifier(svm, train_data_features, train_data[output],
                        test_data_features, test_data[output])


# In[19]:


rf = RandomForestClassifier(n_estimators=100, random_state=rand_seed)

train_test_classifier(rf, train_data_features, train_data[output],
                        test_data_features, test_data[output])


# In[20]:


mlp = MLPClassifier(hidden_layer_sizes=(20,20,20,20), verbose=True, tol=0.001, random_state=rand_seed)
train_test_classifier(mlp, train_data_features, train_data[output],
                        test_data_features, test_data[output])


# In[21]:

#Vertical Bar Chart to compare the performance of all classifiers
classifiers = ['Linear Regression', 'Naive Bayes', 'Linear SVC', 'Random Forest', 'MLP']  
accuracies = [80.1, 79.9, 80.8,79.4,79.2]  

fig, ax = plt.subplots(figsize=(10, 6))  
ax.bar(classifiers, accuracies)
ax.set_xlabel('Classifiers')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of Different Classifiers')
ax.set_ylim(75, 85)
plt.show()



# In[33]:

#Infinite while loop to ask user for input to classifiy, press q to exit
while True:
    input_tweet = input("Tweet To Classify (q to quit): ")
    
    if input_tweet.lower() == 'q':
        break
        
    try:
        input_vectorized = vectorizer.transform([input_tweet])
        prediction = logistic_reg.predict(input_vectorized)
        if prediction[0] == 'pos':
            print("Predicted Sentiment: Positive")
        else:
            print("Predicted Sentiment: Negative")
    except ValueError:
        print("Invalid input. Please enter a valid tweet or 'q' to quit.")
    
print("Thank you for using our program!")

