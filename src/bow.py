import numpy
import pandas

from pathlib import Path
from sklearn import metrics, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


USED_SENTI_METHOD = "vader"


def _generate_test_splits(l: list[tuple[str, float]]) -> tuple:
    # need tuples like: (['foo', 'bar'], 'pos') or (['foo', 'bar'], 'neg')
    test_pairs = [(x[0].split(), "pos" if float(x[1]) > 0.5 else "neg") for x in l]

    train, test = train_test_split(test_pairs, test_size=0.25, random_state=42)
    x_train = [" ".join(words) for (words, _) in train]
    x_test = [" ".join(words) for (words, _) in test]
    y_train = [label for (_, label) in train]
    y_test = [label for (_, label) in test]

    return x_train, x_test, y_train, y_test


def bow_csv(file_path: Path) -> dict[str, float]:
    with open(file_path) as file_in:
        df = pandas.read_csv(file_path)

    # get test splits
    x_train, x_test, y_train, y_test = _generate_test_splits(
        (t for t in zip(df["text"], df["sent_vader"]))
    )

    # vectorize training data
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=2)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    # Mean cross: validation accuracy
    scores = cross_val_score(LogisticRegression(), x_train, y_train, cv=5)

    # Training set score: LogisticRegression
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)

    # Confusion matrix
    confusion = metrics.confusion_matrix(y_test, logreg.predict(x_test))

    # Training set score: RandomForestClassifier
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)

    # Best cross-validation score; Best parameters
    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(x_train, y_train)

    # Support vector machine
    rbf = svm.SVC(kernel="rbf", gamma=0.5, C=0.1).fit(x_train, y_train)
    rbf_pred = rbf.predict(x_test)
    rbf_accuracy = metrics.accuracy_score(y_true=y_test, y_pred=rbf_pred)

    return dict(
        lr_mean_cross=numpy.mean(scores),
        lr_trainin_set_score=logreg.score(x_train, y_train),
        lr_test_set_score=logreg.score(x_test, y_test),
        rf_trainin_set_score=rfc.score(x_train, y_train),
        rf_test_set_score=rfc.score(x_test, y_test),
        confusion_matrix=str(confusion).replace("\n", ","),
        gs_best_cross_validation_score=grid.best_score_,
        gs_best_params=grid.best_params_,
        rbf_accuracy=rbf_accuracy,
    )