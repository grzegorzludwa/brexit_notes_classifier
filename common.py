import numpy as np
import csv, re

from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn import metrics

from nltk.stem import WordNetLemmatizer


def read_comments(doc_file):
    # will store the comments
    X = []
    # will store the sentiment labels
    Y = []

    # open the file, force utf-8 encoding if this isn't the default on your system
    with open(doc_file, encoding="utf-8") as f:
        # read the file line by line
        for line in f:
            try:
                # split the line into the four parts mentioned above
                sentiment, comment = line.strip().split("\t")
            except ValueError as e:
                raise ValueError("Cannot split line by tab char")

            # add the comment and its label to the respective lists
            X.append(comment)
            Y.append(sentiment)

    return X, Y


def read_comments_csv(doc_file, delimiter):
    # will store the comments
    X = []
    # will store the sentiment labels
    Y = []

    # open the file, force utf-8 encoding if this isn't the default on your system
    with open(doc_file, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        # read the file line by line
        for line in reader:
            # split the line into the four parts mentioned above
            sentiment, comment = line

            # add the comment and its label to the respective lists
            X.append(comment)
            Y.append(sentiment)

    return X, Y


def text_preprocess(documents):
    stemmer = WordNetLemmatizer()

    preprocessed_docs = []
    for document in documents:
        # Remove all the special characters
        document = re.sub(r"\W", " ", str(document))
        # remove all single characters
        document = re.sub(r"\s+[a-zA-Z]\s+", " ", document)
        # Remove single characters from the start
        document = re.sub(r"\^[a-zA-Z]\s+", " ", document)
        # Substituting multiple spaces with single space
        document = re.sub(r"\s+", " ", document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r"^b\s+", "", document)
        # Converting to Lowercase
        document = document.lower()
        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = " ".join(document)
        preprocessed_docs.append(document)

    return preprocessed_docs


def extract_features(X, Y, vectorizer):
    X_features = vectorizer.fit_transform(X)

    n = 6000
    best_features_classif = SelectKBest(f_classif, k=n)
    X = best_features_classif.fit_transform(X_features, Y)

    return X


def predict_sentiments(X, vectorizer, trained_classifier):
    X = text_preprocess(X)
    X_features = extract_features(X, Y, vectorizer)

    Y = trained_classifier.predict(X_features)
    return Y


def train_documents_classifier(X, y, features_number=5000):
    vect = TfidfVectorizer(
        ngram_range=(1, 3),
        analyzer="word",
        sublinear_tf=True,
        min_df=1,
        stop_words=stopwords.words("english"),
    )

    best_features_classif = SelectKBest(f_classif, k=features_number)

    clf = ComplementNB(alpha=0.001, class_prior=None, fit_prior=True, norm=False)

    # preprocess documents (remove specia charackters and lemmatize)
    X = text_preprocess(X)
    # learn features vectorizer
    vectorizer = vect.fit(X)
    # extract features
    features = vectorizer.transform(X)
    # get 'features_number' best features
    features = best_features_classif.fit_transform(features, y)

    trained_classifier = clf.fit(features, y)

    print("New trained classifier scores")
    cv_score(features, y, trained_classifier)


def cv_score(X, y, classifier):
    scores = cross_validate(
        classifier,
        X,
        y,
        scoring="accuracy",
        cv=5,
        return_estimator=True,
        return_train_score=True,
    )

    print("\nCross Validate scores:")
    
    train_score = scores["train_score"].mean()
    print("avg. train score:  %0.3f" % train_score)
    
    #print(scores["test_score"])
    test_score = scores["test_score"].mean()
    print("avg. test score:  %0.3f" % test_score)

    train_time = scores["fit_time"].mean()
    print("avg. train time: %0.3fs" % train_time)

    test_time = scores["score_time"].mean()
    print("avg. test time:  %0.3fs" % test_time)

    return scores


def print_scores(Y, Y_pred):
    print("\nClassification report:")
    print(metrics.classification_report(Y, Y_pred))
    print("\nConfusion matrix:")
    print(metrics.confusion_matrix(Y, Y_pred))
    print()
