# imports
import csv
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

UNWANTED_CSV_ROWS = {
    "source",
    "created_at",
    "description",
    "profile_image_url",
    "verified",
    "protected",
    "location",
}


def _clean_text(text: str) -> str:
    # lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(url_re, "", text)

    # remove ATs and hashtags
    # TODO: clear possessive S: e.g. @WeDidItNYC's
    text = re.sub(twitter_at_re, "", text)
    text = re.sub(twitter_hashtag_re, "", text)

    # clear special characters
    text = re.sub(special_characters_re, " ", text)

    # # sentence tokenize
    # text = sent_tokenize(text)

    # word tokenize
    text = word_tokenize(text)

    # drop punctuation, stopwords and numbers
    new_text = []
    for word in text:
        if punctuation_re.match(word):
            continue

        if word in stopwords.words("english") or word in stopwords.words("german"):
            continue

        try:
            float(word)
        except ValueError:
            new_text.append(word)

    return " ".join(new_text)


def _clean_csv(reader: csv.DictReader):
    for row in tqdm(reader):
        row[2] = _clean_text(row[2])
        yield row


def clean_csv(file_path: Path, out_path: Path):
    # load data
    with open(file_path, encoding="utf-8") as file_in:
        reader = csv.reader(file_in, delimiter=",")
        header = next(reader)

        # write cleaned data
        with open(out_path, mode="w", encoding="utf-8") as file_out:
            writer = csv.writer(file_out, delimiter=",")
            writer.writerow(header)
            for cleaned_row in _clean_csv(reader):
                writer.writerow(cleaned_row)
