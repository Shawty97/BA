import csv
from pathlib import Path
from tqdm import tqdm


TOP_WORD_NUMBER = 10


def count_words(file_path: Path) -> dict[str, int]:
    # count words
    with open(file_path, encoding="utf-8") as file_in:
        word_counts = dict()
        reader = csv.reader(file_in, delimiter=",")
        _ = next(reader)

        for row in tqdm(reader, desc="Counting"):
            words = row[2].split(" ")
            for word in words:
                if word not in word_counts:
                    word_counts[word] = 0

                word_counts[word] += 1

    # select and return top 10
    numbers = [word_counts.values()]
    top10_counts = sorted(range(len(numbers)), key=lambda i: numbers[i], reverse=True)
    return top10_counts, numbers
