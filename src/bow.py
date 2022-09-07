import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


USED_SENTI_METHOD = "flair"


def _generate_test_splits(data: dict) -> tuple:
    test_pairs = [
        (t["text"].split(), "pos" if float(t["sentiment"]) > 0.5 else "neg")
        for t in data
    ]

    train, test = train_test_split(test_pairs, test_size=0.25, random_state=42)
    x_train = [" ".join(words) for (words, _) in train]
    x_test = [" ".join(words) for (words, _) in test]
    y_train = [label for (_, label) in train]
    y_test = [label for (_, label) in test]

    return x_train, x_test, y_train, y_test


def _vectorize_data():
    pass


def bow_json(file_path: Path):
    # load data
    with open(file_path, encoding="utf-8") as file_in:
        tweets = json.load(file_in)

    # create Pandas dataframe
    tweets = [
        {
            "tweet_id": d["tweet_id"],
            "sentiment": round(float(d["sentiment_scores"]["vader"])),
            "text": d["text"],
        }
        for d in tweets
    ]
    df = pd.DataFrame.from_dict(data=tweets)

    # get test splits
    x_train, x_test, y_train, y_test = _generate_test_splits(tweets)

    # vectorize training data
    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    feature_names = vectorizer.get_feature_names()

    # regression
    scores = cross_val_score(LogisticRegression(), x_train, y_train, cv=5)
    print(f"Mean cross - validation accuracy: {np.mean(scores):.3f}")

    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    print(f"\nTraining set score: {logreg.score(x_train, y_train):.3f}")
    print(f"Test set score: {logreg.score(x_test, y_test):.3f}")

    confusion = confusion_matrix(y_test, logreg.predict(x_test))
    print("\nConfusion matrix:")
    print(confusion)
    
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    print(f"\nTraining set score: {rfc.score(x_train, y_train):.3f}")
    print(f"Test set score: {rfc.score(x_test, y_test):.3f})")

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(x_train, y_train)
    print(f"\nBest cross-validation score: {grid.best_score_:.2f})")
    print(f"Best parameters: {grid.best_params_}")

    # # write dataframe
    # with open(file_path.parent / 'bowed_tweets.hdf', mode="w", encoding="utf-8") as file_out:
    #     df.to_hdf(file_out)
