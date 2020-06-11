import pandas as pd
import string
import numpy as np
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
            max_features=4000, # Probable best value
            # min_df = 1 > 0.8447
            # min_df=50
        )
        # fit the vectorizer on the text
        vectorizer.fit(self.training['selected_text'])

        # get the vocabulary
        inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
        self.vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]


    def train(self):
	# Classify all reviews into arrays
        self.posReviews = self.training[self.training['sentiment'] == 'positive']
        self.nuReviews = self.training[self.training['sentiment'] == 'neutral']
        self.negReviews = self.training[self.training['sentiment'] == 'negative']

        # Get all words from dataset
        self.posText = [word for tweet in self.posReviews["text"] for word in list(set(self.vocabulary) & set(self.clean_text(tweet).split()))]
        self.nuText = [word for tweet in self.nuReviews["text"] for word in list(set(self.vocabulary) & set(self.clean_text(tweet).split()))]
        self.negText = [word for tweet in self.negReviews["text"] for word in list(set(self.vocabulary) & set(self.clean_text(tweet).split()))]
        self.text = [word for tweet in self.training["text"] for word in list(set(self.vocabulary) & set(self.clean_text(tweet).split()))]

        # Get all words from selected dataset
        self.selectedPosText = [word for tweet in self.posReviews["selected_text"] for word in list(set(self.vocabulary) & set(self.clean_text(tweet).split()))]
        self.selectedNuText = [word for tweet in self.nuReviews["selected_text"] for word in list(set(self.vocabulary) & set(self.clean_text(tweet).split()))]
        self.selectedNegText = [word for tweet in self.negReviews["selected_text"] for word in list(set(self.vocabulary) & set(self.clean_text(tweet).split()))]

        # Get counters for all word in classifications
        self.numReviewsTotal = len(self.training)
        self.probPos = len(self.posReviews) / self.numReviewsTotal
        self.probNu = len(self.nuReviews) / self.numReviewsTotal
        self.probNeg = len(self.negReviews) / self.numReviewsTotal

        # Counters for positive and negative word occurences
        self.posNum = Counter(self.posText)
        self.nuNum = Counter(self.posText)
        self.negNum = Counter(self.negText)

        # Counters for positive and negative word occurences
        self.posSelectedNum = Counter(self.selectedPosText)
        self.nuSelectedNum = Counter(self.selectedPosText)
        self.negSelectedNum = Counter(self.selectedNegText)

        pos_words = {}
        neutral_words = {}
        neg_words = {}

        for k in self.vocabulary:
            pos = self.posNum[k]
            neutral = self.nuNum[k]
            neg = self.negNum[k]

            pos_words[k] = pos/len(self.posText)
            neutral_words[k] = neutral/len(self.nuText)
            neg_words[k] = neg/len(self.negText)

        # We need to account for the fact that there will be a lot of words used in tweets of every sentiment.
        # Therefore, we reassign the values in the dictionary by subtracting the proportion of tweets in the other
        # sentiments that use that word.

        self.neg_words_adj = {}
        self.pos_words_adj = {}
        self.neutral_words_adj = {}

        for key, value in neg_words.items():
            self.neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])

        for key, value in pos_words.items():
            self.pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])

        for key, value in neutral_words.items():
            self.neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])


    def calculate_selected_text(self, df_row, tol = 0):
        offset = 0
        tweet = df_row['text']
        sentiment = df_row['sentiment']

        if(sentiment == 'neutral'):
            return tweet

        elif(sentiment == 'positive'):
            dict_to_use = self.pos_words_adj # Calculate word weights using the pos_words dictionary

        elif(sentiment == 'negative'):
            dict_to_use = self.neg_words_adj # Calculate word weights using the neg_words dictionary

        words = self.clean_text(tweet).split()
        words_len = len(words)
        subsets = [words[i:j+1] for i in range(words_len) for j in range(i,words_len)]

        score = 0
        selection_str = '' # This will be our choice
        lst = sorted(subsets, key = len) # Sort candidates by length


        for i in range(len(subsets)):
            new_sum = 0 # Sum for the current substring

            # Calculate the sum of weights for each word in the substring
            for p in range(len(lst[i])):
                if(lst[i][p] in dict_to_use.keys()):
                    new_sum += dict_to_use[lst[i][p]]

#                print(lst[i][p])
            # If the sum is greater than the score, update our current selection
            if(new_sum > score + tol):
                score = new_sum
                selection_str = lst[i]
            #tol = tol*5 # Increase the tolerance a bit each time we choose a selection

        # If we didn't find good substrings, return the whole text
        if(len(selection_str) == 0):
            selection_str = words

        return ' '.join(selection_str)

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
#        filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
#        filters = '"\'%()+-/:;<=>@[\\]^_`{|}~\t\n'
#        translate_dict = dict((c, " ") for c in filters)
#        translate_map = str.maketrans(translate_dict)
#        text = text.translate(translate_map)

        return text


def main():
    # load = np.load('incorrect_indexes.npy')
    # print(load)
    twitter_train = pd.read_csv('tweet-sentiment-extraction/train.csv', delimiter=',')
    twitter_test = pd.read_csv('tweet-sentiment-extraction/test.csv', delimiter=',')
    twitter_train = twitter_train.dropna()

    sentimentExtract = Sentiment(twitter_train[0:21984], twitter_test)
    sentimentExtract.train()

    sum = 0.0
    # Set up training and testing run throughs
    for index, row in twitter_train[21984:].iterrows():
        # sentimentExtract.extract(row["text"], row["selected_text"], row["sentiment"])
        prediction = sentimentExtract.calculate_selected_text(row, 0.001)
        # if (index > 50):
        #     break
        print("text:", row['text'], "\nselected:", row['selected_text'], "\nprediction: ", prediction)
        print("\n\n\n\n")
        sum += jaccard(row['selected_text'], prediction)
    print(((1/len(twitter_train[21984:]))*sum))


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
