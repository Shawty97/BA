import json

from pathlib import Path

from flair.models import TextClassifier
from flair.data import Sentence
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    def flair(text: str):
        return Sentence(text).to_dict()["labels"][0]["score"]

    def textblob(text: str) -> float:
        return TextBlob(text).sentiment.polarity

    def vader(text: str) -> float:
        return SentimentIntensityAnalyzer().polarity_scores(text)["compound"]


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
        for method in methods:
            for tweet in tweets:
                tweet["sentiment_scores"][method_id] = method(tweet["text"])

    # write analyzed data
    with open(file_path, mode="w", encoding="utf-8") as file_out:
        json.dump(tweets, fp=file_out, indent=2)
