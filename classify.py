#!/usr/bin/env python3
import argparse
import sys
import csv
import pickle

from common import (
    read_comments,
    read_comments_csv,
    cv_score,
    text_preprocess,
    print_scores,
)

from nltk.corpus import stopwords

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from sklearn.pipeline import make_pipeline


parser = argparse.ArgumentParser(
    description="Classify comments sentiments about Brexit. All comments should be categorized as positive/neutral/negative (1/0/-1)"
)
parser.add_argument("test_file", metavar="TEST_FILE", help="Filename with test data")
parser.add_argument(
    "-t",
    "--train",
    metavar="TRAIN_FILE",
    dest="train_file",
    help="Name of file with training data",
)
parser.add_argument(
    "-p",
    "--print_classes",
    action="store_true", 
    help="Print real and predicted classes of test file",
)
parser.add_argument(
    "-c",
    "--classifier",
    default="nb",
    choices=["nb", "svc"],
    help="Chsose classifier during training. nb - Complement Naive Bayes, svc - LinearSVC",
)
parser.add_argument(
    "-s", "--save", action="store_true", help="Overwrite current classifier",
)
parser.add_argument(
    "-n", "--n_of_best", default=5000, type=int, help="Number of best feautures"
)
parser.add_argument(
    "--csv",
    metavar="DELIMETER",
    type=str,
    help="Input data is a csv type. DELIMETER - char between data columns",
)
args = parser.parse_args()


######################################################################
# Main program
print("Loading test data from file: {}".format(args.test_file))

try:
    if args.csv:
        X_test, y_test = read_comments_csv(args.test_file, args.csv)
    else:
        X_test, y_test = read_comments(args.test_file)
except Exception as e:
    print("Bad data format in file")
    print(e)
    exit()

if args.train_file:
    print("Loading train data from file: {}".format(args.train_file))
    try:
        if args.csv:
            X_train, y_train = read_comments_csv(args.train_file, args.csv)
        else:
            X_train, y_train = read_comments(args.train_file)
    except Exception as e:
        print("Bad data format in file")
        print(e)
        exit()

print("Data loaded")

print("Extracting features from data")

if args.train_file:
    tfid_vect = TfidfVectorizer(
        ngram_range=(1, 3),
        analyzer="word",
        sublinear_tf=True,
        min_df=1,
        stop_words=stopwords.words("english"),
    )
    f_classif = SelectKBest(k=args.n_of_best)

    feautures_extractor = make_pipeline(tfid_vect, f_classif)

    X_train = text_preprocess(X_train)
    feautures_train = feautures_extractor.fit_transform(X_train, y_train)

    if args.save:
        pickle.dump(feautures_extractor, open("feautures_extractor.pkl", "wb"))
        print("feautures_extractor saved (overwriten) into file.")
else:
    try:
        feautures_extractor = pickle.load(open("feautures_extractor.pkl", "rb"))
    except Exception as e:
        print("Cannot load vectorizer file")
        print(e)
        exit()
        print("feautures_extractor succesfully loaded from file")


X_test = text_preprocess(X_test)
feautures_test = feautures_extractor.transform(X_test)

print("Features extracted")

if args.train_file:
    print("Training classifier...")

    if args.classifier == "nb":
        classifier = ComplementNB(alpha=0.001, class_prior=None, fit_prior=True, norm=False)
    else:
        classifier = LinearSVC(C=3, penalty="l2", dual=True, loss="hinge", random_state=22)

    classifier = classifier.fit(feautures_train, y_train)

    print("Classifier trained")

    cv_score(feautures_train, y_train, classifier)

    if args.save:
        pickle.dump(classifier, open("classifier.pkl", "wb"))
        print("Classifier saved (overwriten) into file.")
else:
    try:
        classifier = pickle.load(open("classifier.pkl", "rb"))
    except Exception as e:
        print("Cannot load classifier file")
        print(e)
        exit()
    print("Classifier succesfully loaded from file")

print("Prediction process started")
y_predicted = classifier.predict(feautures_test)

print("Prediction scores")
print_scores(y_test, y_predicted)

if args.print_classes:
    for i in range(len(y_predicted)):
        print("{:>4} {:>2} {:>2} {}".format(i, y_test[i], y_predicted[i], args.test_file))
