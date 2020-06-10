import pandas as pd
import math
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import re

DEBUG_VALUES = False
DEBUG_BAYES = False


def main():
    # Importing the dataset
    twitter_train = pd.read_csv('tweet-sentiment-extraction/train.csv', delimiter=',')
    twitter_test = pd.read_csv('tweet-sentiment-extraction/test.csv', delimiter=',')

    # this vectorizer will skip stop words
    vectorizer = CountVectorizer(
        stop_words="english",
        preprocessor=clean_text,
        max_features=4400, # Probable best value
        # min_df = 1 > 0.8447
        # min_df=50
    )
    # fit the vectorizer on the text
    twitter_train = twitter_train.dropna()
    vectorizer.fit(twitter_train['selected_text'])

    # get the vocabulary
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]

    alpha = 1.4

    posText = []
    nuText = []
    negText = []
    text = []

    posReviews, nuReviews, negReviews = [], [], []

    posReviews = twitter_train[twitter_train['sentiment'] == 'positive']
    nuReviews = twitter_train[twitter_train['sentiment'] == 'neutral']
    negReviews = twitter_train[twitter_train['sentiment'] == 'negative']

#    print("Pos Reviews", posReviews)
#    print("Nu Reviews", nuReviews)
#    print("Neg Reviews", negReviews)

    #    posText = [word for tweet in posReviews for word in list(set(vocabulary) & set(clean_text(tweet.split())))]
    posText = [word for tweet in posReviews["text"] for word in list(set(vocabulary) & set(clean_text(tweet).split()))]
    nuText = [word for tweet in nuReviews["text"] for word in list(set(vocabulary) & set(clean_text(tweet).split()))]
    negText = [word for tweet in negReviews["text"] for word in list(set(vocabulary) & set(clean_text(tweet).split()))]

#    print("Pos text", posText)
#    print("nu text", nuText)
#    print("neg text", negText)
#
    numReviewsTotal = len(twitter_train)
    probPos = len(posReviews) / numReviewsTotal
    probNu = len(nuReviews) / numReviewsTotal
    probNeg = len(negReviews) / numReviewsTotal

    # Counters for positive and negative word occurences
    posNum = Counter(posText)
    nuNum = Counter(posText)
    negNum = Counter(negText)

    # # Debug print statements
    if (DEBUG_VALUES):
        print("Total reviews: ", numReviewsTotal)
        print("Prob pos: ", probPos)
        print("Prob nu: ", probNu)
        print("Prob neg: ", probNeg)
        print("Positive Text: ", posNum)
        print("Negative Text: ", negNum)

    # Set up training and testing run throughs
    f = open("results.txt", "w")

    for data in twitter_test:
        # Run through the data for tuning
        correct, total = 0, 0
        for i, review in enumerate(data):
            print("Testing #:", i + 1)
            print(review)

            sentiment = classify(review[0],
                                 numReviewsTotal,
                                 probPos,
                                 probNu,
                                 probNeg,
                                 posNum,
                                 nuNum,
                                 negNum,
                                 posText,
                                 nuText,
                                 negText,
                                 vocabulary,
                                 alpha
                                 )

            # Total number of correct
            if(sentiment == review[1]):
                print("Correct")
                correct += 1

            else:
                print("Negative")

            total += 1

        print (data[1], "Accuracy:", correct / total)
        f.write(data[1] + " Accuracy: " + str(correct / total) + "\n")
    f.write("Training data that was used was from file \'twitter_trainSet.txt\' and twitter_test data was from file \'testSet.txt\'\n")
    f.close()

def classify(review, numTotalReviews, probPos, probNu, probNeg, posNum, nuNum, negNum, posText, nuText, negText, vocabulary, alpha):
    # For each word in the cleaned text of the review
    pos, nu, neg = 0.0, 0.0, 0.0
    for word in clean_text(review).split():
        # getting values for bayes equation
        vocabLen = len(vocabulary)


        # Bayes for positivity
        probWordPos = ((posNum[word] + alpha) / (len(posText) + (vocabLen * alpha)))
        posNumerator = (probWordPos * probPos)
        pos += math.log(posNumerator)

        # Bayes for neutrality
        probWordNu = ((nuNum[word] + alpha) / (len(nuText) + (vocabLen * alpha)))
        nuNumerator = (probWordNu * probNu)
        nu += math.log(posNumerator)

        # Bayes for negativity
        probWordNeg = ((negNum[word] + alpha) / (len(negText) + (vocabLen * alpha)))
        negNumerator = (probWordNeg * probNeg)
        neg += math.log(negNumerator)


        if (DEBUG_BAYES):
            print("pos numerator", posNumerator)
            print("pos denominator", posDenominator)
            print("pos num/den", (posNumerator / posDenominator))

            print("neg numerator", negNumerator)
            print("neg denominator", negDenominator)
            print("neg num/den", (negNumerator / negDenominator))

            print("word", word)
            print("pos:", pos)
            print("neg:", neg)

    # Return values for positive, nuetral, or negative
    if (pos > nu) and (pos > neg):
        return "positive"
    elif (nu > neg) and (nu > pos):
        return "neutral"
    else:
        return "negative"




def clean_text(text):
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


if __name__ == "__main__":
    main()
