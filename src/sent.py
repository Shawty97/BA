import csv
from pathlib import Path

from flair.data import Sentence
from flair.models import TextClassifier
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DECIMAL_PLACES = 3

sia = TextClassifier.load("en-sentiment")
sid = SentimentIntensityAnalyzer()


class SentimentAnalyzer:
    @staticmethod
    def flair(text: str) -> float:
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


def sentiment_csv(file_path: Path, out_path: Path):
    # load data
    with open(file_path, encoding="utf-8") as file_in:
        reader = csv.reader(file_in, delimiter=",")
        header = next(reader)

        # write cleaned data
        with open(out_path, mode="w", encoding="utf-8", newline="") as file_out:
            writer = csv.writer(file_out, delimiter=",")
            writer.writerow(header + ["sent_flair", "sent_textblob", "sent_vader"])
            sa = SentimentAnalyzer

            for row in reader:
                if not row:
                    continue

                writer.writerow(
                    row + [sa.flair(row[2]), sa.textblob(row[2]), sa.vader(row[2])]
                )
