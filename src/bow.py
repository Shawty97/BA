from pathlib import Path
from datetime import datetime

import numpy
import pandas
from tqdm import tqdm
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

current_step = 0


def _update_script_step(text: str, pbar: tqdm):
    global current_step
    current_step += 1

    pbar.set_description(text)
    pbar.update(current_step)


def _generate_test_splits(l: list[tuple[str, float]]) -> tuple[numpy.array]:
    # need tuples like: (['foo', 'bar'], 'pos') or (['foo', 'bar'], 'neg')
    test_pairs = numpy.array(
        [(x[0].split(), "pos" if float(x[1]) > 0.5 else "neg") for x in l]
    )

    train, test = train_test_split(test_pairs, test_size=0.25, random_state=42)
    x_train = numpy.array([" ".join(words) for (words, _) in train])
    x_test = numpy.array([" ".join(words) for (words, _) in test])
    y_train = numpy.array([label for (_, label) in train])
    y_test = numpy.array([label for (_, label) in test])

    return x_train, x_test, y_train, y_test


def bow_csv(file_path: Path) -> dict[str, float]:
    print("loading data")
    df = pandas.read_csv(file_path, encoding="utf-8")

    with tqdm(total=7) as pbar:
        _update_script_step("get test splits", pbar)
        x_train, x_test, y_train, y_test = _generate_test_splits(
            (t for t in zip(df["text"], df["sent_vader"]))
        )

        _update_script_step("vectorize training data", pbar)
        vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=2)
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)

        _update_script_step("Mean cross: validation accuracy", pbar)
        scores = cross_val_score(LogisticRegression(), x_train, y_train, cv=5)

        _update_script_step("Training set score: LogisticRegression", pbar)
        logreg = LogisticRegression()
        logreg.fit(x_train, y_train)
        lr_confusion = metrics.confusion_matrix(y_test, logreg.predict(x_test))

        _update_script_step("Training set score: RandomForestClassifier", pbar)
        rfc = RandomForestClassifier()
        rfc.fit(x_train, y_train)
        rf_confusion = metrics.confusion_matrix(y_test, rfc.predict(x_test))

        _update_script_step("Best cross-validation score; Best parameters", pbar)
        param_grid = {"C": [0.001, 0.01, 0.1, 1, 10]}
        grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
        grid.fit(x_train, y_train)

        # _update_script_step("Support vector machine", pbar)
        # rbf = svm.SVC(kernel="rbf", gamma=0.5, C=0.1).fit(x_train, y_train)
        # rbf_pred = rbf.predict(x_test)
        # rbf_accuracy = metrics.accuracy_score(y_true=y_test, y_pred=rbf_pred)

        return dict(
            lr_mean_cross=numpy.mean(scores),
            lr_trainin_set_score=logreg.score(x_train, y_train),
            lr_test_set_score=logreg.score(x_test, y_test),
            lr_confusion_matrix=str(lr_confusion).replace("\n", ","),
            rf_trainin_set_score=rfc.score(x_train, y_train),
            rf_test_set_score=rfc.score(x_test, y_test),
            rf_confusion_matrix=str(rf_confusion).replace("\n", ","),
            gs_best_cross_validation_score=grid.best_score_,
            gs_best_params=grid.best_params_,
            # rbf_accuracy=rbf_accuracy,
        )
