# NLP_based_Text_Classification
  
  
Text-based unstructured data is prevalent today and can be found in chats, emails, social media posts, and survey results. Text can be a valuable source of information, but because it is unstructured, it can be challenging to draw conclusions from it.
One of the crucial tasks in supervised machine learning is text classification (ML). It is a method of classifying documents into tags or categories that enables us to swiftly and efficiently organize and analyze material. One of the core NLP tasks, it has numerous applications including sentiment analysis, spam detection, topic labeling, intent detection, etc.

**Let’s divide the classification problem into the below steps:**

1. Setup: Importing Libraries
2. Loading the data set & Exploratory Data Analysis
3. Text pre-processing
4. Extracting vectors from text (Vectorization)
5. Running ML algorithms
6. Conclusion



**Step 2: Loading the data set & EDA**

The data set that we will be using for this article is the famous “Natural Language Processing with Disaster Tweets” data set where we’ll be predicting whether a given tweet is about a real disaster (target=1) or not (target=0)
In this project, you’re challenged to build a machine learning model that predicts which Tweets are about real disasters and which ones aren’t. You’ll have access to a dataset of 10,000 tweets that were hand classified.
We have 7,613 tweets in training (labelled) dataset and 3,263 in the test(unlabelled) dataset

Exploratory Data Analysis (EDA)
1. Class distribution: There are more tweets with class 0 ( no disaster) than class 1 ( disaster tweets). We can say that the dataset is relatively balanced with 4342 non-disaster tweets (57%) and 3271 disaster tweets (43%). Since the data is balanced, we won’t be applying data-balancing techniques like SMOTE while building the model

2. Missing values: We have ~2.5k missing values in location field and 61 missing values in keyword column

3. Number of words in a tweet: Disaster tweets are more wordy than the non-disaster tweets
The average number of words in a disaster tweet is 15.17 as compared to an average of 14.7 words in a non-disaster tweet

4. Number of characters in a tweet: Disaster tweets are longer than the non-disaster tweets
The average characters in a disaster tweet is 108.1 as compared to an average of 95.7 characters in a non-disaster tweet

**Step 3: Text Pre-Processing**

Before we move to model building, we need to preprocess our dataset by removing punctuations & special characters, cleaning texts, removing stop words, and applying lemmatization
Simple text cleaning processes: Some of the common text cleaning process involves:
- Removing punctuations, special characters, URLs & hashtags
- Removing leading, trailing & extra white spaces/tabs
- Typos, slangs are corrected, abbreviations are written in their long forms

1. Stop-word removal: We can remove a list of generic stop words from the English vocabulary using nltk. A few such words are ‘i’,’you’,’a’,’the’,’he’,’which’ etc.
2. Stemming: Refers to the process of slicing the end or the beginning of words with the intention of removing affixes(prefix/suffix)
3. Lemmatization: It is the process of reducing the word to its base form


**Step 4: Extracting vectors from text (Vectorization)**

It’s difficult to work with text data while building Machine learning models since these models need well-defined numerical data. The process to convert text data into numerical data/vector, is called vectorization or in the NLP world, word embedding. Bag-of-Words(BoW) and Word Embedding (with Word2Vec) are two well-known methods for converting text data to numerical data.
There are a few versions of Bag of Words, corresponding to different words scoring methods. We use the Sklearn library to calculate the BoW numerical values using these approaches:
1. Count vectors: It builds a vocabulary from a corpus of documents and counts how many times the words appear in each document
2. Term Frequency-Inverse Document Frequencies (tf-Idf): Count vectors might not be the best representation for converting text data to numerical data. So, instead of simple counting, we can also use an advanced variant of the Bag-of-Words that uses the term frequency–inverse document frequency (or Tf-Idf). Basically, the value of a word increases proportionally to count in the document, but it is inversely proportional to the frequency of the word in the corpus

Word2Vec: One of the major drawbacks of using Bag-of-words techniques is that it can’t capture the meaning or relation of the words from vectors. Word2Vec is one of the most popular technique to learn word embeddings using shallow neural network which is capable of capturing context of a word in a document, semantic and syntactic similarity, relation with other words, etc.

We can use any of these approaches to convert our text data to numerical form which will be used to build the classification model. With this in mind, I am going to first partition the dataset into training set (80%) and test set (20%)

**Step 5. Running ML algorithms**

It’s time to train a machine learning model on the vectorized dataset and test it. I demonstrated the basics of building a text classification model comparing Bag-of-Words (with Tf-Idf) and Word Embedding with Word2Vec with the following ML algorithms.
1. Logistic Regression
   Confusion matix for tf-idf Logistic Regression.
   Accuracy 0.795
   ![image](https://user-images.githubusercontent.com/61462986/224088480-29307295-9861-40fd-8ddf-7be929b51e75.png)

2. Random Forest
   Confusion matix for w2v Random Forest.
   Accuracy 0.676
   ![image](https://user-images.githubusercontent.com/61462986/224088868-a77f0acd-ac5b-4fe3-8c2f-bfdfb7712339.png)

3. XGBoost
   Confusion matix for w2v Random Forest.
   Accuracy 0.764
   ![image](https://user-images.githubusercontent.com/61462986/224089218-a0ebf865-065b-4be7-90ca-bc5b5e22129b.png)



  
  
  
  
  
  
  
