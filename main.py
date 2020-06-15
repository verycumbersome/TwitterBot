import pandas as pd
from transformers import *
import tokenizers
import string
import emoji
import torch
import transformers as ppb # pytorch transformersimport random
import numpy as np
import math
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.preprocessing.sequence import pad_sequences
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

        self.idf = TfidfVectorizer(
            stop_words="english",
            max_features=10000, # Probable best value
        )

        # fit the cv on the text
        self.cv.fit(self.training['text'])
        self.idf.fit(self.training['text'])

        # Initialize BERT model
        model_class = ppb.DistilBertModel
        tokenizer_class =  ppb.DistilBertTokenizer
        pretrained_weights = 'distilbert-base-uncased'

        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)

        tokenized = self.training["text"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

        print(tokenized.str.pad(15, side ='left'))

        padded = pad_sequences(tokenized)

        input_ids = torch.tensor(np.array(padded))

        with torch.no_grad():
            last_hidden_states = model(input_ids)

         # Slice the output for the first position for all the sequences, take all hidden unit outputs
        features = last_hidden_states[0][:,0,:].numpy()

    def calculate_tuples(self, sentiment, dRow):
        tweets = self.training[self.training['sentiment'] == sentiment]
        tuples = []

        # Get all words from dataset
        for tweet in tweets[dRow]:
                cleanTweet = self.clean_text(tweet, True).split()
                for index, word in enumerate(cleanTweet):
                    if word not in self.cv.vocabulary_.keys():
                        if (index == len(cleanTweet) - 1):
                            continue

                        #print("Word/clean:", word, cleanTweet[index + 1])
                        tuples.append((word, cleanTweet[index + 1]))

        return tuples


    def train(self):
        # Get all words from dataset
        posText = self.training[self.training['sentiment'] == 'positive']
        nuText = self.training[self.training['sentiment'] == 'neutral']
        negText = self.training[self.training['sentiment'] == 'negative']

        posNum = self.cv.transform(posText['text'])
        nuNum = self.cv.transform(nuText['text'])
        negNum = self.cv.transform(negText['text'])

        posIdfNum = self.idf.transform(posText['text'])
        nuIdfNum = self.idf.transform(nuText['text'])
        negIdfNum = self.idf.transform(negText['text'])

        posNum = pd.DataFrame(posNum.toarray(), columns=self.cv.get_feature_names())
        nuNum = pd.DataFrame(nuNum.toarray(), columns=self.cv.get_feature_names())
        negNum = pd.DataFrame(negNum.toarray(), columns=self.cv.get_feature_names())

        posIdfNum = pd.DataFrame(posIdfNum.toarray(), columns=self.idf.get_feature_names())
        nuIdfNum = pd.DataFrame(nuIdfNum.toarray(), columns=self.idf.get_feature_names())
        negIdfNum = pd.DataFrame(negIdfNum.toarray(), columns=self.idf.get_feature_names())

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
            pos_words[k] = posNum[k].sum()
            neutral_words[k] = nuNum[k].sum()
            neg_words[k] = negNum[k].sum()

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

        # convert text to lowercase
        text = text.strip().lower()
        text = emoji.demojize(text)

        emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)


        if (punc):
            # replace punctuation characters with spaces
            filters = '"\'%()+-/:;<=>@[\\]^_`{|}~\t\n'
            translate_dict = dict((c, " ") for c in filters)
            translate_map = str.maketrans(translate_dict)
            text = text.translate(translate_map)

        return text

def main():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.save_vocabulary('.')

    MAX_LEN = 96
    tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file='vocab.json',
            merges_file='merges.txt',
            lowercase=True,
            add_prefix_space=True
            )
    sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

    twitter_train = pd.read_csv('./kaggle/input/tweet-sentiment-extraction/train.csv', delimiter=',')
    twitter_test = pd.read_csv('./kaggle/input/tweet-sentiment-extraction/test.csv', delimiter=',')
    twitter_train = twitter_train.dropna()

    sentimentExtract = Sentiment(twitter_train[0:21984])
    sentimentExtract.train()

    # sum = 0.0
    # # Set up training and testing run throughs
    # for index, row in twitter_train[21984:].iterrows():
        # prediction = sentimentExtract.calculate_selected_text(row, 0.001)
        # # if (index > 10):
            # # break
        # print("text:", row['text'], "\nselected:", row['selected_text'], "\nprediction: ", prediction)
        # print("\n\n\n\n")
        # sum += jaccard(row['selected_text'], prediction)
    # print(((1/len(twitter_train[21984:]))*sum))


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
