import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import string
import nltk
from nltk.corpus import stopwords

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

from sklearn import metrics

#Uploading the dataset file containing the messages
messages = pd.read_csv('spam.csv',encoding = 'latin-1')
messages.head()

messages.tail()


#Preprocessing, Removing columns that are unnamed
messages = messages.drop(labels = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
messages.columns = ["label", "message"]


messages.info()


messages.describe()
#use groupby to use describe by label, this way we can begin to think about the features that separate ham and spam!

messages.groupby('label').describe().T

messages['length'] = messages['message'].apply(len)
messages.head()


# Count the frequency of top 5 messages.
messages['message'].value_counts().rename_axis(['message']).reset_index(name='counts').head()

messages["label"].value_counts().plot(kind = 'pie',explode=[0, 0.1],figsize=(6, 6),autopct='%1.1f%%',shadow=True)
plt.title("Spam vs Not Spam")
plt.legend(["Not Spam", "Spam"])
plt.show()
fig = plt.figure()
fig.savefig('plot.png')

plt.figure(figsize=(12,6))
messages['length'].plot(bins=100, kind='hist') # with 100 length bins (100 length intervals) 
plt.title("Frequency Distribution of Message Length")
plt.xlabel("Length")
plt.ylabel("Frequency")


messages['length'].describe()

 #Using Masking To Find the the message having length Of 910 words
 
messages[messages['length'] == 910]['message'].iloc[0]


messages.hist(column='length', by='label', bins=50,figsize=(12,4))


#
#Basic Preprocessing of the dataset for spam and not spam messages for further classification 
#

def text_preprocess(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    nopunc = nopunc.lower()
    
    # Now just remove any stopwords and non alphabets
    nostop=[word for word in nopunc.split() if word.lower() not in stopwords.words('english') and word.isalpha()]
    return nostop

spam_messages = messages[messages["label"] == "spam"]["message"]
ham_messages = messages[messages["label"] == "ham"]["message"]
print("No of spam messages : ",len(spam_messages))
print("No of Non spam messages : ",len(ham_messages))
#Wordcloud for Spam Messages¶
# This may take a while....
spam_words = text_preprocess(spam_messages)

# lets print some spam words
spam_words[:10]




spam_wordcloud = WordCloud(width=600, height=400).generate(' '.join(spam_words))
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


print("Top 10 Spam words are :\n")
print(pd.Series(spam_words).value_counts().head(10))



#Wordcloud for Non Spam Messages

ham_words = text_preprocess(ham_messages)


#print some Non Spam words

ham_words[:10]


ham_wordcloud = WordCloud(width=600, height=400).generate(' '.join(ham_words))
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(ham_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


messages.head()

# remove punctuations/ stopwords from all SMS 
messages["message"] = messages["message"].apply(text_preprocess)


# Conver the SMS into string from list
messages["message"] = messages["message"].agg(lambda x: ' '.join(map(str, x)))

messages["message"][7]



# Creating the Bag of Words


vectorizer = CountVectorizer()
bow_transformer = vectorizer.fit(messages['message'])

print("20 Bag of Words (BOW) Features: \n")
print(vectorizer.get_feature_names()[20:40])

print("\nTotal number of vocab words : ",len(vectorizer.vocabulary_))



message4 = messages['message'][3]
print(message4)


# fit_transform : Learn the vocabulary dictionary and return term-document matrix.
bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)



messages_bow = bow_transformer.transform(messages['message'])


print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)


tfidf4 = tfidf_transformer.transform(bow4)

messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)


#Lets convert our clean text into a representation that a machine learning model can understand. I'll use the Tfifd for this



from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")
features = vec.fit_transform(messages["message"])
print(features.shape)

print(len(vec.vocabulary_))

msg_train, msg_test, label_train, label_test = \
train_test_split(messages_tfidf, messages['label'], test_size=0.2)


print("train dataset features size : ",msg_train.shape)
print("train dataset label size", label_train.shape)

print("\n")

print("test dataset features size", msg_test.shape)
print("test dataset lable size", label_test.shape)



from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

clf = MultinomialNB()
spam_detect_model = clf.fit(msg_train, label_train)


predict_train = spam_detect_model.predict(msg_train)


print("Classification Report \n",metrics.classification_report(label_train, predict_train))
print("\n")
print("Confusion Matrix \n",metrics.confusion_matrix(label_train, predict_train))
print("\n")
print("Accuracy of Train dataset : {0:0.3f}".format(metrics.accuracy_score(label_train, predict_train)))



print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', messages['label'][3])



label_predictions = spam_detect_model.predict(msg_test)
print(label_predictions)

print(metrics.classification_report(label_test, label_predictions))
print(metrics.confusion_matrix(label_test, label_predictions))



# Printing the Overall Accuracy of the model
print("Accuracy of the model : {0:0.3f}".format(metrics.accuracy_score(label_test, label_predictions)))


#Applying second calssification model
#Classification through Support Vector Machine SVM

from sklearn.svm import SVC


X_train, X_test, y_train, y_test = train_test_split( messages_tfidf, messages['label'], test_size=0.2, random_state=0)

classifier = SVC(kernel = 'rbf', random_state = 10)
classifier.fit(X_train, y_train)
print(classifier.score(X_test,y_test))


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
spam_detect_model = clf.fit(msg_train, label_train)
from yellowbrick.classifier import ClassificationReport
classes = ["Spam", "Not Spam"]

classifier = ClassificationReport(spam_detect_model,classes=classes, support=True)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

#Classification by logistic regression
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


