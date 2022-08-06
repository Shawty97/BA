# imports
import json
import re
import string
from pathlib import Path

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# paths
curent_directory = Path(__file__).parent
data_source = curent_directory / "data" / "tweets.json"
cleaned_data_target = curent_directory / "data" / "tweets_cleaned.json"

# punctuation regex
punctuation_re = re.compile(f"[{re.escape(string.punctuation)}]")

# load data
with open(data_source, encoding="utf-8") as file_in:
    tweets = json.load(file_in)["tweets"]

for tweet in tweets:
    # lowercase
    tweet["text"] = tweet["text"].lower()

    # word tokenize
    tweet["text"] = word_tokenize(tweet["text"])

    # sentence tokenize
    try:
        tweet["text"] = sent_tokenize(tweet["text"])
    except TypeError as e:
        print(
            "Sentence Tokenization failed for tweet {tweet_id} ({exception})".format(
                tweet_id=tweet["tweet_id"],
                exception=e,
            )
        )
        # TODO: what should we do here? stop for this tweet? continue with the remaining steps?

    # punctuation
    tweet["text"] = [word for word in tweet["text"] if punctuation_re.sub("", word)]

    # stopwords
    tweet["text"] = [
        word for word in tweet["text"] if word not in stopwords.words("english")
    ]

# write cleaned data
with open(cleaned_data_target, mode="w", encoding="utf-8") as file_out:
    json.dump(tweets, fp=file_out, indent=2)
