from pathlib import Path
from fire import Fire

from clean import clean_json
from sentiment import sentiment_json
from bow import bow_json


class CLI:
    def all(self, file_path: Path):
        """
        Clean and analyze the sentiment of the tweets, then apply BoW.
        """
        self.clean(file_path=file_path)
        self.sent(file_path=file_path, method="all")
        self.bow(file_path=file_path)

    def clean(self, file_path: Path):
        """
        Clean the tweets
        (Warning: Make a backup of the file if you want to keep it)
        """
        clean_json(file_path=file_path)

    def sent(self, file_path: Path, method: str = "all"):
        """
        Analyze the sentiment of the tweets.
        (Runs ALL methods sequentually if no method is specified)
        """
        sentiment_json(file_path=file_path, method_id=method)

    def bow(self, file_path: Path):
        """
        Apply "Bag of Words" to tweets
        """
        bow_json(file_path=file_path)


if __name__ == "__main__":
    Fire(CLI)
