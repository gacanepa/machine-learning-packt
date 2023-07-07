# Import the libraries we will use
import csv
import numpy as np
import pandas
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob


def read_dataset():
    # Load the SMSSpamCollection training dataset
    with open("SMSSpamCollection") as sms_spam_collection:
        training_dataset = [line.strip() for line in sms_spam_collection]
        print(f"The training dataset has a total of {len(training_dataset)} records")
    # Pre-process the dataset and add column names (class for spam or ham, message for the text)
    raw_data = pandas.read_csv(
        "SMSSpamCollection",
        sep="\t",
        quoting=csv.QUOTE_NONE,
        names=["class", "message"],
    )
    # First 5 messages
    print(raw_data.head())

    # Classification
    print(raw_data.groupby("class").count())

    return raw_data


def words_into_base_form(message_words):
    message = str(message_words).lower()
    words_base = TextBlob(message).words
    return [word.lemma for word in words_base]


# Entry point for the application
def main():
    data = read_dataset()

    training_vector = CountVectorizer(analyzer=words_into_base_form).fit(
        data.get("message")
    )

    # Let's see the classification of the 10th message
    message10 = training_vector.transform([data.get("message")[9]])
    print(message10)

    # Bag of words from the entire dataset
    bag_of_words = training_vector.fit_transform(data.get("message").values)

    # Weight of words (term frequency and inverse document frequency)
    messages_tf_idf = TfidfTransformer().fit(bag_of_words).transform(bag_of_words)

    # Train the model
    spam_detector = MultinomialNB().fit(messages_tf_idf, data.get("class").values)

    # Play with variations of the messages found in the training dataset
    example = ["My mom loves me and I love her"]

    result = spam_detector.predict(training_vector.transform(example))[0]

    print(f"The message '{example[0]}' has been classified as {result}")


if __name__ == "__main__":
    main()
