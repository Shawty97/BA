from collections import Counter
from pathlib import Path
import pandas

def count_words(file_path: Path) -> Counter:
    with open(file_path) as file_in:
        df = pandas.read_csv(file_in)

        counter = Counter()
        for text in df["text"]:
            counter += Counter(text.split(' '))

        return counter
