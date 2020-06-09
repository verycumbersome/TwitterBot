import pandas as pd
import math
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import re

# Importing the dataset
training = []
with open("trainingSet.txt", 'r') as file:
    line = file.readline()
    i = 0
    while line:
        training.append(line.split('\t'))
        line = file.readline()
        training[i][1] = int(training[i][1][1])
        i += 1

testing = []
with open("testSet.txt", 'r') as file:
    line = file.readline()
    i = 0
    while line:
        testing.append(line.split('\t'))
        testing[i][1] = int(testing[i][1][1])
        line = file.readline()
        i += 1

DEBUG_VALUES = False
DEBUG_BAYES = False


def main():
    # this vectorizer will skip stop words
    vectorizer = CountVectorizer(
        stop_words="english",
        preprocessor=clean_text,
        max_features=4400, # Probable best value
        # min_df = 1 > 0.8447
        # min_df=50
    )

    # fit the vectorizer on the text
    vectorizer.fit([i[0] for i in training])

    # get the vocabulary
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]

    alpha = 1.4

    posText = []
    negText = []
    text = []

    posReviews, negReviews = [], []
    for index, review in enumerate(training):
        if (review[1]):
#            print("Negative")
            for word in list(set(vocabulary) & set(clean_text(review[0]).split())):
                posText.append(word)
                text.append(word)

            posReviews.append(review[0])

        else:
#            print("Positive")
            for word in list(set(vocabulary) & set(clean_text(review[0]).split())):
                negText.append(word)
                text.append(word)

            negReviews.append(review[0])


    numReviewsTotal = len(training)
    probPos = len(posReviews) / numReviewsTotal
    probNeg = len(negReviews) / numReviewsTotal

    # Counters for positive and negative word occurences
    posNum = Counter(posText)
    negNum = Counter(negText)


    # # Debug print statements
    if (DEBUG_VALUES):
        print("Total reviews: ", numReviewsTotal)
        print("Prob pos: ", probPos)
        print("Prob neg: ", probNeg)
        print("Positive Text: ", posNum)
        print("Negative Text: ", negNum)

    # Set up training and testing run throughs
    f = open("results.txt", "w")
    
    curData = [(training, "Training"), (testing, "Testing")]
    for data in curData:
        # Run through the data for tuning
        correct, total = 0, 0
        for i, review in enumerate(data[0]):
            print("Testing #:", i + 1)

            sentiment = classify(review[0],
                numReviewsTotal,
                probPos,
                probNeg,
                posNum,
                negNum,
                posText,
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
    f.write("Training data that was used was from file \'trainingSet.txt\' and testing data was from file \'testSet.txt\'\n")
    f.close()

def classify(review, numTotalReviews, probPos, probNeg, posNum, negNum, posText, negText, vocabulary, alpha):
    # For each word in the cleaned text of the review
    pos, neg = 0.0, 0.0
    for word in clean_text(review).split():
        # getting values for bayes equation
        vocabLen = len(vocabulary)


        # Bayes for positivity
        probWordPos = ((posNum[word] + alpha) / (len(posText) + (vocabLen * alpha)))
        posNumerator = (probWordPos * probPos)
        pos += math.log(posNumerator)


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

    if pos > neg:
        # print("positive")
        return 1
    else:
        # print("negative")
        return 0




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
