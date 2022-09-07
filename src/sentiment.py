import json

from pathlib import Path

from flair.models import TextClassifier
from flair.data import Sentence
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DECIMAL_PLACES = 3

sia = TextClassifier.load("en-sentiment")
sid = SentimentIntensityAnalyzer()


class SentimentAnalyzer:
    @staticmethod
    def flair(text: str):
        """returns a float between 0 and 1"""
        sentence = Sentence(text)
        sia.predict(sentence)
        return sentence.labels[0]._score

    @staticmethod
    def textblob(text: str) -> float:
        """returns a float between -1 and 1"""
        return TextBlob(text).sentiment.polarity

    @staticmethod
    def vader(text: str) -> float:
        """returns a float between -1 and 1"""
        return sid.polarity_scores(text)["compound"]


def sentiment_json(file_path: Path, method_id: str = "all"):
    # select which methods to run
    if method_id == "all":
        methods = [
            getattr(SentimentAnalyzer, m)
            for m in dir(SentimentAnalyzer)
            if not m.startswith("_")
        ]
    else:
        methods = [getattr(SentimentAnalyzer, method_id)]

    # load data
    with open(file_path, encoding="utf-8") as file_in:
        tweets = json.load(file_in)

        # make sentiment analysis and add it to the attributes
        for tweet in tweets:
            tweet["sentiment_scores"] = {}

            sentence = " ".join(tweet["text"])

            for method in methods:
                tweet["sentiment_scores"][method.__name__] = "{:.2f}".format(method(sentence))

    # write analyzed data
    with open(file_path, mode="w", encoding="utf-8") as file_out:
        json.dump(tweets, fp=file_out, indent=2)
