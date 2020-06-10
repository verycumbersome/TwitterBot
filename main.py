import pandas as pd
import math
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import re


class Sentiment():
    def __init__(self, training, testing):
        # Importing the dataset
        self.training = training
        self.testing = testing

        # this vectorizer will skip stop words
        vectorizer = CountVectorizer(
            stop_words="english",
            preprocessor=self.clean_text,
            max_features=4400, # Probable best value
            # min_df = 1 > 0.8447
            # min_df=50
        )
        # fit the vectorizer on the text
        self.training = self.training.dropna()
        vectorizer.fit(self.training['selected_text'])

        # get the vocabulary
        inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
        self.vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]

        alpha = 1.4

    def train(self):
        self.posText = []
        self.nuText = []
        self.negText = []
        self.text = []

        self.posReviews = []
        self.nuReviews = []
        self.negReviews = []

        self.posReviews = self.training[self.training['sentiment'] == 'positive']
        self.nuReviews = self.training[self.training['sentiment'] == 'neutral']
        self.negReviews = self.training[self.training['sentiment'] == 'negative']


        self.selectedPosText = [word for tweet in self.posReviews["selected_text"] for word in list(set(self.vocabulary) & set(self.clean_text(tweet).split()))]
        self.selectedNuText = [word for tweet in self.nuReviews["selected_text"] for word in list(set(self.vocabulary) & set(self.clean_text(tweet).split()))]
        self.selectedNegText = [word for tweet in self.negReviews["selected_text"] for word in list(set(self.vocabulary) & set(self.clean_text(tweet).split()))]

        self.posText = [word for tweet in self.posReviews["text"] for word in list(set(self.vocabulary) & set(self.clean_text(tweet).split()))]
        self.nuText = [word for tweet in self.nuReviews["text"] for word in list(set(self.vocabulary) & set(self.clean_text(tweet).split()))]
        self.negText = [word for tweet in self.negReviews["text"] for word in list(set(self.vocabulary) & set(self.clean_text(tweet).split()))]


        self.numReviewsTotal = len(self.training)
        self.probPos = len(self.posReviews) / self.numReviewsTotal
        self.probNu = len(self.nuReviews) / self.numReviewsTotal
        self.probNeg = len(self.negReviews) / self.numReviewsTotal

        # Counters for positive and negative word occurences
        self.posNum = Counter(self.posText)
        self.nuNum = Counter(self.posText)
        self.negNum = Counter(self.negText)


    def classify(self, text, sentiment):
        # For each word in the cleaned text of the review
        if (sentiment == "positive"):
            print("positive")
        elif (sentiment == "neutral"):
            print("neutral")
        else:

            print("negative")


    def clean_text(self, text):
        # remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # remove the characters [\], ['] and ["]
        text = re.sub(r"\\", "", text)
        text = re.sub(r"\'", "", text)
        text = re.sub(r"\"", "", text)
        #pattern = r'[^a-zA-z0-9\s]'
        #text = re.sub(pattern, '', text)

        # convert text to lowercase
        text = text.strip().lower()

        # replace punctuation characters with spaces
        filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        translate_dict = dict((c, " ") for c in filters)
        translate_map = str.maketrans(translate_dict)
        text = text.translate(translate_map)

        return text


def main():
    twitter_train = pd.read_csv('tweet-sentiment-extraction/train.csv', delimiter=',')
    twitter_test = pd.read_csv('tweet-sentiment-extraction/test.csv', delimiter=',')

    sentimentExtract = Sentiment(twitter_train, twitter_test)
    sentimentExtract.train()

    # Set up training and testing run throughs
    print(twitter_train["text"][1])
    sentiment = twitter_train["sentiment"][1]
    sentimentExtract.classify(twitter_train["text"][1], twitter_train["sentiment"][1])


def eval(truth, pred):
    n = len(truth)
    sum = 0.0
    for i in range(n):
        sum += jaccard(truth[i], pred[i])
    return ((1/n)*sum)


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

    
if __name__ == "__main__":
    main()
