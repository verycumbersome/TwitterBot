import pandas as pd
import string
import random
import numpy as np
import math
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re


class Sentiment():
    def __init__(self, training):
        # Importing the dataset
        self.training = training

        # this cv will skip stop words
        self.cv = CountVectorizer(
            stop_words="english",
            max_features=10000, # Probable best value
        )
        # fit the cv on the text
        self.cv.fit(self.training['text'])

        self.idf = TfidfVectorizer(
            stop_words="english",
            max_features=10000, # Probable best value
        )

        # print idf values
        # self.idf = pd.DataFrame(self.cv.idf_, index=self.cv.get_feature_names(), columns=["idf_weights"])


        # self.posNum = pd.DataFrame(self.cv.idf_, index=self.cv.get_feature_names(), columns=["idf_weights"])


    def calculate_tuples(self, sentiment, dRow, isTuple=1):
        tweets = self.training[self.training['sentiment'] == sentiment]
        tuples = []

        # Get all words from dataset
        for tweet in tweets[dRow]:
            if (isTuple):
                cleanTweet = self.clean_text(tweet, True).split()
                for index, word in enumerate(cleanTweet):
                    if word not in self.cv.vocabulary_.keys():
                        if (index == len(cleanTweet) - 1):
                            continue

                        #print("Word/clean:", word, cleanTweet[index + 1])
                        tuples.append((word, cleanTweet[index + 1]))
            else:
                cleanTweet = list(set(self.cv.vocabulary_.keys()) & set(self.clean_text(tweet).split()))
                for word in cleanTweet:
                    tuples.append(word)

        return tuples


    def train(self):
        # Get all words from dataset
        posText = self.training[self.training['sentiment'] == 'positive']
        nuText = self.training[self.training['sentiment'] == 'neutral']
        negText = self.training[self.training['sentiment'] == 'negative']

        posNum = self.cv.transform(posText['text'])
        nuNum = self.cv.transform(nuText['text'])
        negNum = self.cv.transform(negText['text'])

        posNum = pd.DataFrame(posNum.toarray(), columns=self.cv.get_feature_names())
        nuNum = pd.DataFrame(nuNum.toarray(), columns=self.cv.get_feature_names())
        negNum = pd.DataFrame(negNum.toarray(), columns=self.cv.get_feature_names())

        print(posNum.sum())

        # print(self.cv.vocabulary_)
        # Get all words from dataset
        self.posTuples = self.calculate_tuples("positive", "text")
        self.nuTuples = self.calculate_tuples("neutral", "text")
        self.negTuples = self.calculate_tuples("negative", "text")

        # Get all words from selected dataset
        self.selectedPosTuples = self.calculate_tuples("positive", "selected_text")
        self.selectedNuTuples = self.calculate_tuples("neutral", "selected_text")
        self.selectedNegTuples = self.calculate_tuples("negative", "selected_text")

        # Counters for positive and negative word occurences
        self.posNumTuples = Counter(self.posTuples)
        self.nuNumTuples = Counter(self.nuTuples)
        self.negNumTuples = Counter(self.negTuples)

        # Counters for positive and negative word occurences
        self.posSelectedNumTuples = Counter(self.selectedPosTuples)
        self.nuSelectedNumTuples = Counter(self.selectedNuTuples)
        self.negSelectedNumTuples = Counter(self.selectedNegTuples)


        self.allTuples = {**self.posNumTuples, **self.nuNumTuples, **self.negNumTuples}

        pos_words = {}
        neutral_words = {}
        neg_words = {}

        pos_tuples = {}
        neutral_tuples = {}
        neg_tuples = {}

        for k in self.cv.vocabulary_.keys():
            pos = posNum[k].sum()
            neutral = nuNum[k].sum()
            neg = negNum[k].sum()

            pos_words[k] = pos/len(posText)
            neutral_words[k] = neutral/len(nuText)
            neg_words[k] = neg/len(negText)

        for t in self.allTuples:
            posTuples = self.posSelectedNumTuples[t]
            nuTuples = self.nuSelectedNumTuples[t]
            negTuples = self.negSelectedNumTuples[t]

            pos_tuples[t] = posTuples/len(self.posTuples)
            neutral_tuples[t] = nuTuples/len(self.nuTuples)
            neg_tuples[t] = negTuples/len(self.negTuples)


        # We need to account for the fact that there will be a lot of words used in tweets of every sentiment.
        # Therefore, we reassign the values in the dictionary by subtracting the proportion of tweets in the other
        # sentiments that use that word.

        self.neg_words_adj = {}
        self.pos_words_adj = {}
        self.neutral_words_adj = {}

        self.neg_tuples_adj = {}
        self.pos_tuples_adj = {}
        self.neutral_tuples_adj = {}

        for key, value in neg_words.items():
            self.neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])

        for key, value in pos_words.items():
            self.pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])

        for key, value in neutral_words.items():
            self.neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])

        for key, value in neg_tuples.items():
            self.neg_tuples_adj[key] = neg_tuples[key] - (neutral_tuples[key] + pos_tuples[key])

        for key, value in pos_tuples.items():
            self.pos_tuples_adj[key] = pos_tuples[key] - (neutral_tuples[key] + neg_tuples[key])

        for key, value in neutral_tuples.items():
            self.neutral_tuples_adj[key] = neutral_tuples[key] - (neg_tuples[key] + pos_tuples[key])


    def calculate_selected_text(self, df_row, tol = 0):
        tweet = df_row['text']
        sentiment = df_row['sentiment']

        if(sentiment == 'neutral'):
            return tweet

        elif(sentiment == 'positive'):
            dict_to_use = self.pos_words_adj # Calculate word weights using the pos_words dictionary
            tuple_dict = self.pos_tuples_adj

        elif(sentiment == 'negative'):
            dict_to_use = self.neg_words_adj # Calculate word weights using the neg_words dictionary
            tuple_dict = self.neg_tuples_adj

        words = self.clean_text(tweet).split()
        words_len = len(words)
        subsets = [words[i:j+1] for i in range(words_len) for j in range(i,words_len)]

        score = 0
        selection_str = '' # This will be our choice
        lst = sorted(subsets, key = len) # Sort candidates by length


        for i in range(len(subsets)):
            new_sum = 0 # Sum for the current substring

            # Calculate the sum of weights for each word in the substring
            if (len(lst[i]) <= 1):
                for p in range(len(lst[i])):
                    if(lst[i][p] in dict_to_use.keys()):
                        new_sum += dict_to_use[lst[i][p]]

            else:
                for p in range(len(lst[i])):
                    if(lst[i][p] in dict_to_use.keys()):
                        new_sum += dict_to_use[lst[i][p]]

                    if (p + 1) >= len(lst[i]):
                        continue

                    if((lst[i][p],lst[i][p+1]) in tuple_dict.keys()):
                        new_sum += 0.575 * tuple_dict[(lst[i][p],lst[i][p+1])]

            # If the sum is greater than the score, update our current selection
            if(new_sum > score + tol):
                score = new_sum
                selection_str = lst[i]
            #tol = tol*5 # Increase the tolerance a bit each time we choose a selection

        # If we didn't find good substrings, return the whole text
        if(len(selection_str) == 0):
            selection_str = words

        return ' '.join(selection_str)


    def clean_text(self, text, punc=False):
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

        if (punc):
            # replace punctuation characters with spaces
            filters = '"\'%()+-/:;<=>@[\\]^_`{|}~\t\n'
            translate_dict = dict((c, " ") for c in filters)
            translate_map = str.maketrans(translate_dict)
            text = text.translate(translate_map)

        return text

def main():
    twitter_train = pd.read_csv('./kaggle/input/tweet-sentiment-extraction/train.csv', delimiter=',')
    twitter_test = pd.read_csv('./kaggle/input/tweet-sentiment-extraction/test.csv', delimiter=',')
    twitter_train = twitter_train.dropna()

    sentimentExtract = Sentiment(twitter_train[0:21984])
    sentimentExtract.train()

    sum = 0.0
    # Set up training and testing run throughs
    for index, row in twitter_train[21984:].iterrows():
        prediction = sentimentExtract.calculate_selected_text(row, 0.001)
        # if (index > 1000):
            # break
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
