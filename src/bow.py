import numpy
import pandas

from pathlib import Path
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


USED_SENTI_METHOD = "flair"


def _generate_test_splits(l: list[tuple[str, float]]) -> tuple:
    # need tuples like: (['foo', 'bar'], 'pos') or (['foo', 'bar'], 'neg')
    test_pairs = [(x[0].split(), "pos" if float(x[1]) > 0.5 else "neg") for x in l]

    train, test = train_test_split(test_pairs, test_size=0.25, random_state=42)
    x_train = [" ".join(words) for (words, _) in train]
    x_test = [" ".join(words) for (words, _) in test]
    y_train = [label for (_, label) in train]
    y_test = [label for (_, label) in test]

    return x_train, x_test, y_train, y_test


def bow_vidhya() -> float:
    """
    Sentiment Analysis Using Python
    https://www.analyticsvidhya.com/blog/2022/07/sentiment-analysis-using-python/
    """

    with open(Path(__file__).parent / "data" / "tweets_analyzed.csv") as file_in:
        df = pandas.read_csv(file_in)

    # Pre-Processing and Bag of Word Vectorization using Count Vectorizer
    token = RegexpTokenizer(r"[a-zA-Z0-9]+")
    cv = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        tokenizer=token.tokenize,
    )
    text_counts = cv.fit_transform(df["text"])

    # Splitting the data into trainig and testing
    x_train, x_test, y_train, y_test = train_test_split(
        text_counts, df["sent_vader"], test_size=0.25, random_state=5
    )

    # Training the model
    multi_nb = MultinomialNB()
    multi_nb.fit(x_train, y_train)

    # Parameter tuning
    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(
        estimator=multi_nb,
        param_grid=param_grid,
        cv=5,
    )
    grid.fit(x_train, y_train)

    # Caluclating the accuracy score of the model
    predicted = multi_nb.predict(x_test)
    accuracy_score = metrics.accuracy_score(predicted, y_test)
    return accuracy_score


def bow_csv(file_path: Path) -> dict[str, float]:
    with open(file_path) as file_in:
        df = pandas.read_csv(file_path)

    # get test splits
    x_train, x_test, y_train, y_test = _generate_test_splits(
        (t for t in zip(df["text"], df["sent_vader"]))
    )

    # vectorize training data
    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    # TODO: find out what we wanna use this for
    feature_names = vectorizer.get_feature_names()

    # regression
    scores = cross_val_score(LogisticRegression(), x_train, y_train, cv=5)
    # print(f"Mean cross - validation accuracy: {np.mean(scores):.3f}")

    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    # print(f"\nTraining set score: {logreg.score(x_train, y_train):.3f}")
    # print(f"Test set score: {logreg.score(x_test, y_test):.3f}")

    confusion = metrics.confusion_matrix(y_test, logreg.predict(x_test))
    # print("\nConfusion matrix:")
    # print(confusion)

    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    # print(f"\nTraining set score: {rfc.score(x_train, y_train):.3f}")
    # print(f"Test set score: {rfc.score(x_test, y_test):.3f})")

    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(x_train, y_train)
    # print(f"\nBest cross-validation score: {grid.best_score_:.2f})")
    # print(f"Best parameters: {grid.best_params_}")

    return dict(
        lr_mean_cross=numpy.mean(scores),
        lr_trainin_set_score=logreg.score(x_train, y_train),
        lr_test_set_score=logreg.score(x_test, y_test),
        rf_trainin_set_score=rfc.score(x_train, y_train),
        rf_test_set_score=rfc.score(x_test, y_test),
        confusion_matrix=str(confusion).replace("\n", ","),
        gs_best_cross_validation_score=grid.best_score_,
        gs_best_params=grid.best_params_,
    )
