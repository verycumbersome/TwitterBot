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
        segments = self.training["selected_text"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

        print(self.training["text"][1], tokenized[1])
        print(self.training["selected_text"][1], segments[1])

        tokenized_ids = tokenized.apply((lambda x: tokenizer.convert_tokens_to_ids(x)))
        segments_id = segments.apply((lambda x: tokenizer.convert_tokens_to_ids(x)))

        tokens_tensor = torch.tensor(np.array(pad_sequences(tokenized))).to(torch.int64)
        segments_tensors = torch.tensor(np.array(pad_sequences(segments))).to(torch.int64)

        # Set the model in evaluation mode to deactivate the DropOut modules
        # This is IMPORTANT to have reproducible results during evaluation!
        model.eval()

        # If you have a GPU, put everything on cuda
        try:
            tokens_tensor = tokens_tensor.to('cuda')
            segments_tensors = segments_tensors.to('cuda')
            model.to('cuda')

        except:
            print("ERROR: Must have graphics card with CUDA compatibility. Running without GPU acceleration")

        # Predict all tokens
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            predictions = outputs[0]

        # confirm we were able to predict 'henson'
        predicted_index = torch.argmax(predictions[0, masked_index]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        assert predicted_token == 'henson'


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


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))



if __name__ == "__main__":
    main()
