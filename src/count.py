from collections import Counter
from pathlib import Path
import pandas

def count_words(file_path):
    with open(file_path) as file_in:
        df = pandas.read_csv(file_path)

        counter = Counter()
        for text in df["text"]:
            counter += Counter(text.split(' '))

        return counter.most_common(10)

print(count_words('./src/data/tweets.csv'))