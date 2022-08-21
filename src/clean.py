# imports
import json
import re
import string
from pathlib import Path

from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# regex
punctuation_re = re.compile(rf"[{re.escape(string.punctuation)}]")
url_re = re.compile(r"https?://\S+")
twitter_at_re = re.compile(r"[@][\w_]+")
twitter_hashtag_re = re.compile(r"[#][\w]+")
special_characters_re = re.compile(r"[^\w\d\s\-]+")

def _clean_tweets(tweets: list[dict]):
    for tweet in tqdm(tweets):
        # lowercase
        tweet["text"] = tweet["text"].lower()

        # remove URLs
        tweet["text"] = re.sub(url_re, "", tweet["text"])

        # remove ATs and hashtags
        # TODO: clear possessive S: e.g. @WeDidItNYC's
        tweet["text"] = re.sub(twitter_at_re, "", tweet["text"])
        tweet["text"] = re.sub(twitter_hashtag_re, "", tweet["text"])

        # clear special characters
        tweet["text"] = re.sub(special_characters_re, " ", tweet["text"])

        # word tokenize
        tweet["text"] = word_tokenize(tweet["text"])

        # sentence tokenize
        try:
            tweet["text"] = sent_tokenize(tweet["text"])
        except TypeError as e:
            pass

        # drop punctuation, stopwords and numbers
        new_text = []
        for word in tweet["text"]:
            if punctuation_re.match(word):
                continue

            if word in stopwords.words("english"):
                continue

            try:
                float(word)
            except ValueError:
                new_text.append(word)

        tweet["text"] = new_text


def clean_json(file_path: Path):
    # load data
    with open(file_path, encoding="utf-8") as file_in:
        tweets = json.load(file_in)["tweets"]

    # clean
    _clean_tweets(tweets)

    # write cleaned data
    with open(file_path, mode="w", encoding="utf-8") as file_out:
        json.dump(tweets, fp=file_out, indent=2)
