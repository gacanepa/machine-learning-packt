# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:49:46 2016

@author: amnesia
"""

# Import libraries
import csv
from textblob import TextBlob
import pandas
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# Load the training dataset 'SMSSpamCollection' into variable 'data'
data = [line.rstrip() for line in open('SMSSpamCollection')]

# Print number of messages
print ''
print 'The training dataset has a total of', len(data), 'records'
print ''

"""
Read the dataset. Specify the field separator is a tab instead of a comma.
Additionally, add column captions ('class' and 'message') for the two fields in the dataset.
To preserve quotations in messages, use QUOTE_NONE.
"""
data = pandas.read_csv('SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["class", "message"])

# Convert each word into its base form
def WordsIntoBaseForm(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]

# Convert each message into a vector
trainingVector = CountVectorizer(analyzer=WordsIntoBaseForm).fit(data['message'])

# View occurrence of words in an arbitrary vector. Use 9 for vector #10.
message10 = trainingVector.transform([data['message'][9]])
print message10
print ''

# Print message #10 for comparison
print data['message'][9]
print ''
# Identify repeated words
print 'First word that appears twice:', trainingVector.get_feature_names()[3437]
print ''
print 'Word that appears three times:', trainingVector.get_feature_names()[5192]
print ''

# Bag-of-words for the entire training dataset
messagesBagOfWords = trainingVector.fit_transform(data['message'].values)

# Weight of words in the entire training dataset - Term Frequency and Inverse Document Frequency
messagesTfidf = TfidfTransformer().fit(messagesBagOfWords).transform(messagesBagOfWords)

# Train the model
spamDetector = MultinomialNB().fit(messagesTfidf, data['class'].values)

# Test message
example = ['England v Macedonia - dont miss the goals/team news. Txt ENGLAND to 99999']

# Result
checkResult = spamDetector.predict(trainingVector.transform(example))[0]

print 'The message [',example[0],'] has been classified as', checkResult
