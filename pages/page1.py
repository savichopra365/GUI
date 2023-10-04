#!/usr/bin/env python
# coding: utf-8


from sklearn import preprocessing
from sklearn.naive_bayes import BernoulliNB
import sklearn
from sklearn import svm, datasets
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import nltk
from plotly import tools
from plotly.offline import iplot
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from collections import defaultdict
from matplotlib.patches import Patch
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from wordcloud import WordCloud
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
from textblob import TextBlob
import matplotlib.pyplot as plt
import string
import re
import streamlit as st


# In[1]:


import pandas as pd
import numpy as np


# In[2]:


st.subheader("Importing the Dataset")

df = pd.read_csv(
    "C:\\Users\\ACER\\Downloads\\archive (3)\\Musical_instruments_reviews.csv")


# In[3]:


reviews = df.copy()


# In[4]:


reviews.isnull().sum()


# In[5]:


# filling the missing review values
cleaned_rev = reviews.dropna()


# In[6]:


cleaned_rev


# In[7]:


cleaned_rev.isnull().sum()
# no null values is left hence we removed all the missing value from data


# In[8]:


cleaned_rev['overall'].value_counts()


# # we can see that the above dataset is highly imbalanced we cannot train our dataset on the imbalanced dataset

# In[9]:


# defining a function
def new_func(row):
    if row['overall'] == 3:
        val = 'neutral'
    elif row['overall'] == 1 or row['overall'] == 2:
        val = 'negative'

    elif row['overall'] == 4 or row['overall'] == 5:
        val = 'positive'
    else:
        val = -1
    return val


# In[10]:


cleaned_rev['sentiment'] = cleaned_rev.apply(new_func, axis=1)


# In[11]:


cleaned_rev.head()


# In[12]:


cleaned_rev['sentiment'].value_counts()


# In[13]:


# extracting the info from date column
new_date = cleaned_rev['reviewTime'].str.split(',', n=1, expand=True)
cleaned_rev['date'] = new_date[0]
cleaned_rev['year'] = new_date[1]
cleaned_rev = cleaned_rev.drop(['reviewTime'], axis=1)

cleaned_rev.head()


# In[14]:


# In[15]:


# splitting bthe date
new_month = cleaned_rev['date'].str.split(' ', n=1, expand=True)
cleaned_rev['month'] = new_month[0]
cleaned_rev['day'] = new_month[1]
cleaned_rev = cleaned_rev.drop(['date'], axis=1)


cleaned_rev.head()


# the date is sucessfully extracted as month year and day

# # punctuation cleaning

# In[16]:


cleaned_rev.head()


# In[17]:


cleaned_Review = cleaned_rev.copy()


# In[18]:


# In[19]:


def review_cleaning(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub('<.*?>+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\w*\d\w*', ' ', text)
    return text


# import re
# import string
#
# def review_cleaning(text):
#     '''Make text lowercase, remove text in square brackets, remove links, remove punctuation,
#     and remove words containing numbers.'''
#     text = str(text).lower()
#     text = re.sub('\[.*?\]', ' ', text)
#     text = re.sub('https?://\S+|www\.\S+', ' ', text)
#     text = re.sub('<.*?>+', ' ', text)
#     text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
#     text = re.sub('\w*\d\w*', ' ', text)
#     return text
#

# In[20]:


cleaned_rev['reviewText'] = cleaned_rev['reviewText'].apply(
    lambda x: review_cleaning(x))


# # remove stop words
# making a list of custom stopwords that do not affect if these words are removed for example if word like"not" is removed sentiment score will not be as accuarate
#  we create a list of custom stop words

# In[21]:


# custom stop words
stop_words = ['yourselves', 'between', 'whom', 'itself', 'is', "she's", 'up', 'herself', 'here', 'your', 'each',
              'we', 'he', 'my', "you've", 'having', 'in', 'both', 'for', 'themselves', 'are', 'them', 'other',
              'and', 'an', 'during', 'their', 'can', 'yourself', 'she', 'until', 'so', 'these', 'ours', 'above',
              'what', 'while', 'have', 're', 'more', 'only', "needn't", 'when', 'just', 'that', 'were', "don't",
              'very', 'should', 'any', 'y', 'isn', 'who',  'a', 'they', 'to', 'too', "should've", 'has', 'before',
              'into', 'yours', "it's", 'do', 'against', 'on',  'now', 'her', 've', 'd', 'by', 'am', 'from',
              'about', 'further', "that'll", "you'd", 'you', 'as', 'how', 'been', 'the', 'or', 'doing', 'such',
              'his', 'himself', 'ourselves',  'was', 'through', 'out', 'below', 'own', 'myself', 'theirs',
              'me', 'why', 'once',  'him', 'than', 'be', 'most', "you'll", 'same', 'some', 'with', 'few', 'it',
              'at', 'after', 'its', 'which', 'there', 'our', 'this', 'hers', 'being', 'did', 'of', 'had', 'under',
              'over', 'again', 'where', 'those', 'then', "you're", 'i', 'because', 'does', 'all']


# In[22]:


cleaned_rev['reviewText'] = cleaned_rev['reviewText'].apply(
    lambda x: " ".join([word for word in x.split() if word not in stop_words]))


# In[23]:


cleaned_rev = cleaned_rev.drop(
    ['unixReviewTime', 'summary', 'reviewerID'], axis='columns')


# The `word for word in x.split()` is a construct used in Python's list comprehension. It's a concise way to iterate through each word in a string and perform some operation on them. Let's break down the components:
#
# - `x`: This represents a string that you want to split into words and iterate through.
#
# - `.split()`: This method is called on the string `x` and splits it into a list of words. By default, it splits the string based on spaces.
#
# - `word for word in ...`: This is the syntax for a list comprehension. It iterates through the list of words generated by `x.split()`. For each word, it assigns the word to the variable `word` and performs an operation on it.
#
# Putting it all together, the construct `word for word in x.split()` generates a sequence of words from the string `x` and iterates through them. This can be useful for various purposes, such as filtering, transforming, or analyzing text data word by word.
#
# For example, let's say you have the string `x = "Hello world, how are you?"`. If you use the construct `word for word in x.split()`, it will generate a sequence of words: `['Hello', 'world,', 'how', 'are', 'you?']`.
#
# You can then use this sequence of words within a list comprehension to perform operations like filtering out specific words, converting them to lowercase, or applying any other transformations you might need.

# In[24]:


def rem_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


# what is list comprehension in python
# Certainly! In Python, a list comprehension is a concise way to create lists by applying an expression to each item in an iterable (like a list, tuple, or range) and optionally applying a filter condition. It's a more compact and readable alternative to using traditional for loops.
#
# The basic structure of a list comprehension is as follows:
#
# ```
# new_list = [expression for item in iterable if condition]
# ```
#
# Here's a breakdown of the components:
#
# 1. **expression**: This is the operation or calculation you want to perform on each item in the iterable to create the new list element.
#
# 2. **item**: It represents each individual item from the iterable that will be processed by the expression.
#
# 3. **iterable**: This is the collection of items you're iterating over, such as a list, tuple, or range.
#
# 4. **condition** (optional): You can include an optional condition that filters the items before they are processed by the expression. Only items that satisfy the condition will be included in the new list.
#
# Here's an example to illustrate:
#
# ```python
# numbers = [1, 2, 3, 4, 5]
# squared_even_numbers = [x**2 for x in numbers if x % 2 == 0]
# ```
#
# In this example, the list comprehension generates a new list `squared_even_numbers` containing the squares of even numbers from the `numbers` list.
#
# List comprehensions can also be nested, and you can use them to create more complex structures like dictionaries or sets. However, it's important to maintain clarity and avoid excessive complexity for readability.
#
# Overall, list comprehensions are a powerful tool in Python for creating lists in a concise and efficient manner.

# In[25]:


cleaned_rev['reviewText'] = cleaned_rev['reviewText'].apply(rem_stopwords)


# In[26]:


# # plotting and visualization

# In[27]:


st.subheader("Years vs Sentiment Count")
st.text('In this plot we will see how many reviews were posted based on sentiments')
st.text("in each year from 2004 to 2014")


cleaned_rev.groupby(['year', 'sentiment'])[
    'sentiment'].count().unstack().plot(legend=True)
plt.xlabel('year')
plt.ylabel('sentiment count')
plt.title('comparison of sentiment with years')
plt.show()

st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("---")


# In[28]:


# ###### insights that we draw from the graph is that as the year passes by we have more hike in positive reviews as comapred to the negative reviews

# In[29]:


cleaned_rev['polarity'] = cleaned_rev['reviewText'].map(
    lambda text: TextBlob(text).sentiment.polarity)
cleaned_rev['review_len'] = cleaned_rev['reviewText'].astype(str).apply(len)
cleaned_rev['word_count'] = cleaned_rev['reviewText'].apply(
    lambda x: len(str(x).split()))


# In[30]:


correlation_matrix = cleaned_rev.corr(numeric_only=True)


# In[31]:


# Create a heatmap of the correlation matrix

st.subheader("Correlation Matrix")

st.text("A correlation matrix is a table that shows the correlation coefficients between ")
st.text(" multiple variables. Correlation coefficients quantify the strength and direction of ")
st.text(" a linear relationship between two variables.")

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("---")


# In[32]:


# In[33]:


# ## comparing all the variables that have a numeric values
# in general we assume that the overall rating score should be proportional to the polarity but the corelation between them
#   is not that strong.  Hence we cannot say the polarity is dependent variable of overall but this is not the truth as we analyse using the corelation matrix
#

# In[34]:

st.subheader("WordCloud")

st.text("A word cloud is a visual representation of a collection of words, where the size of ")
st.text("each word is proportional to its frequency or importance within the text. In other")
st.text(" words, the more frequently a word appears in the text, the larger and more prominent")
st.text(" it appears in the word cloud.")

text = cleaned_rev["reviewText"]
wordcloud = WordCloud(
    width=3000,
    height=2000,
    background_color='black',
    stopwords=STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize=(40, 30),
    facecolor='k',
    edgecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("---")


# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
#
# Example text data (replace with your own text data)
# text_data = cleaned_rev['reviewText']
#
#  Generate the word cloud
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(str(text_data))
#
#  Access the word frequencies
# word_frequencies = wordcloud.words_
#
#  Convert the word frequencies to a list of keywords
# keywords = list(word_frequencies.keys())
#
#  Display the word cloud using Matplotlib
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()
#
#  Display the list of keywords
# print("List of Keywords:", keywords)
#

# In[35]:


# Generate word cloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white').generate(str(text))


st.subheader("Frequency Bar Graph")
st.text("The below bar graph shows the top 5 words with highest normalised frequency ")
st.text("of occurence")


# Access the word frequencies
word_frequencies = wordcloud.words_

# Get the top N words and their frequencies
top_n = 5  # Change this to the number of top words you want
top_words = list(word_frequencies.keys())[:top_n]
top_frequencies = [word_frequencies[word] for word in top_words]

# Create a bar plot of top word frequencies
plt.figure(figsize=(10, 5))
plt.bar(top_words, top_frequencies)
plt.xlabel('Words')
plt.ylabel('Normalised Frequency')
plt.title(f'Top {top_n} Word Frequencies')
plt.show()

st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("---")


# ###### the frequency of words is normalized in wordcloud so that visual representation can be clear and the bar graph also shows the normalized frequency so that it is easy to compare

# In[36]:


label_encoder = LabelEncoder()


# Fit and transform the sentiment column
cleaned_rev['sentiment_encoded'] = label_encoder.fit_transform(
    cleaned_rev['sentiment'])


# In[37]:


cleaned_rev['review_len'].max()


# In[38]:


st.subheader("Scatter Plot")
st.text("it is a graphical representation used to display the relationship between two")
st.text("continous variables and is used to show how to variables are related ")
st.text("")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=cleaned_rev, x='review_len',
                y='sentiment_encoded', hue='sentiment', alpha=0.7)
plt.yticks(list(label_encoder.transform(label_encoder.classes_)),
           label_encoder.classes_)
plt.xlabel('Review Length')
plt.ylabel('Sentiment Encoded')
plt.title('Scatter Plot of Sentiment Encoded vs Text Length')
plt.grid(True)

st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("---")


# In[39]:


cleaned_rev['sentiment'].value_counts()


# In[40]:


cleaned_rev['sentiment_encoded'].value_counts()


# In[41]:


st.set_option('deprecation.showPyplotGlobalUse', False)


# Display the plot


# ###### the graph is not the correct choice for visualization
# Data Range: If the values in cleaned_rev['sentiment_encoded'] are outside the expected range for categorical data, they might be treated as numerical values, leading to issues like negative values on the x-axis.
#
#

# ###### In summary, Patch objects in Matplotlib provide a way to create customized graphical elements that can be used as legend handles. This is particularly useful when you need to represent complex or custom elements in your legend that go beyond simple lines and markers. By using Patch objects, you can enhance the clarity and accuracy of your legends in data visualizations.
#
#
#
#
#

# We have a list of sentiment categories (sentiments) and their corresponding colors (colors).
# Simulated data (data_values) represents the heights of the bars for each sentiment category.
# A bar plot is created using plt.bar().
# The legend handles are created using Patch objects. Each Patch has a specific color and label, corresponding to the sentiment category and its color.
# The plt.legend() function is used to add the custom legend handles to the plot.
# When you run this code, you'll get a bar plot with a legend that uses colored rectangles (custom shapes) to represent the sentiment categories. This showcases how Patch objects can be used to create more informative and visually appealing legends in your data visualizations.

# In the code provided, the zip() function is used to combine two or more iterables (in this case, colors and sentiments) element-wise. It pairs elements from each iterable together, creating tuples where the elements at corresponding positions are grouped together.
#
# In this line of code, zip(colors, sentiments) combines the colors list and the sentiments list element-wise. For each iteration, it takes one color from the colors list and one sentiment from the sentiments list and creates a tuple. The loop then uses these tuples to create Patch objects for each legend handle.
#
#

# In[ ]:


# In[42]:


review_pos = cleaned_rev[cleaned_rev["sentiment"] == 'positive'].dropna()
review_neu = cleaned_rev[cleaned_rev["sentiment"] == 'neutral'].dropna()
review_neg = cleaned_rev[cleaned_rev["sentiment"] == 'negative'].dropna()


## custom function for ngram generation ##
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(
        " ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## custom function for horizontal bar chart ##


def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["review_len"].values[::-1],
        x=df["word_count"].values[::-1],
        showlegend=False,
        orientation='h',
        marker=dict(
            color=color,
        ),
    )
    return trace


## Get the bar chart from positive reviewText ##
freq_dict = defaultdict(int)
for sent in review_pos["reviewText"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["review_len", "word_count"]  # Corrected column names
trace0 = horizontal_bar_chart(fd_sorted.head(25), 'green')

## Get the bar chart from neutral reviewText ##
freq_dict = defaultdict(int)
for sent in review_neu["reviewText"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["review_len", "word_count"]  # Corrected column names
trace1 = horizontal_bar_chart(fd_sorted.head(25), 'grey')

## Get the bar chart from negative reviewText ##
freq_dict = defaultdict(int)
for sent in review_neg["reviewText"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["review_len", "word_count"]  # Corrected column names
trace2 = horizontal_bar_chart(fd_sorted.head(25), 'red')


# Creating two subplots


# In[43]:


cleaned_rev['sentiment_encoded'].value_counts()


# In[44]:


review_features = cleaned_rev.copy()


# In[45]:


review_features = review_features[['reviewText']].reset_index(drop=True)


# In[46]:


# In[47]:


ps = PorterStemmer()

corpus = []

for i in range(0, len(review_features)):
    review = re.sub('[^a-zA-Z]', ' ', review_features['reviewText'][i])
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stop_words]
    review = ' '.join(review)
    corpus.append(review)


# In[48]:


# So, the line of code replaces all characters that are not letters with spaces in the 'reviewText' of the review_features DataFrame at index i. This type of preprocessing is often used to remove punctuation, digits, and other non-alphabetic characters from text data before performing further analysis or natural language processing tasks.

# # TFIDF Vectorizer

# In[49]:


# In[50]:


tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(2, 2))

X = tfidf_vectorizer.fit_transform(review_features['reviewText'])


# In[51]:


y = cleaned_rev['sentiment']


# In[52]:


cleaned_rev['sentiment'].value_counts()


# In[56]:


# # smote

# In[5]:


# In[59]:


# In[ ]:
